import os
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification,BertModel
import torch
import math
from tqdm import tqdm
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from src.utils import *
import torch.nn.functional as F

class ModelBert(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = 'xlm-roberta-base',
            device: str = "cpu",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        if model_name_or_path == None:
            self.tokenizer = None
            self.model = None
        else:
            self.model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
            self.model = self.model.to(self.device)

    def text_to_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

class ModelRerank(nn.Module):
    def __init__(
        self, 
        model_name_or_path = "BAAI/bge-reranker-base",
        device = "cpu"
    ):
        super().__init__()
        self.model_name_or_path  = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        if model_name_or_path != None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.model.to(self.device)
                
    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        val_dataloader:DataLoader = None,
        optimizer_name:str = "AdamW",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        eval_interval:int = 10,
        save_checkpoint_step:int = 10,
        save_model_dir:str = "models",
        first_trainning = True
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location = self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % eval_interval == 0:
                    if val_dataloader != None:
                        self.eval_model(epoch, val_dataloader, log_path)
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )
   
    def forward(self, input:dict, is_train = True):
        output = {}
        batch_size, group_size, _ = input["input_ids"].shape
        batch_data = {
            "input_ids":input["input_ids"].reshape(-1,input["input_ids"].shape[2]).to(self.device),
            "attention_mask":input["attention_mask"].reshape(-1,input["attention_mask"].shape[2]).to(self.device)
        }
        label = input["label"].reshape(-1).to(self.device)
        outputs = self.model(**batch_data)
        scores = outputs["logits"].reshape(batch_size, group_size)
        output["predict"] = scores
        output["label"] = label
        if is_train == False:
            predict_class = torch.argmax(scores,dim = 1)
            output["predict_class"] = predict_class
        return output
    
    # def inference(query:str, doc:str):
        
    
    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer, clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f}, average_hit_rate:{:.4f}, use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg
    
    def test_epoch(self,epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg
    
    def eval_model(self, epoch, val_dataloader, log_path = None):
        self.eval()
        self.to(self.device)
        true_class = []
        predict_class = []
        with torch.no_grad():
            for batch_id, inputs in enumerate(val_dataloader):
                """ forward """
                output = self.forward(inputs,is_train = False)
                true_class.extend(output["label"].cpu().tolist())
                predict_class.extend(output["predict_class"].cpu().tolist())
        
        log_message, accuracy, precision, recall, f1 = calculate_metrics(predict_class, true_class)
        log_message = "Eval Epoch: {:d}\n".format(epoch) + log_message
        print(log_message)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(log_message+"\n")
        return log_message, accuracy, precision, recall, f1
                
    def compute_loss(self, input:dict):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = self.criterion(input["predict"],input["label"])
        else:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = self.criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output
    
    def load_pretrained(self, save_model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(save_model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_model_dir)
        self.model.to(self.device)

    def save_pretrained(self,  save_model_dir):
        self.model.save_pretrained(save_model_dir) 
        self.tokenizer.save_pretrained(save_model_dir)

    def rerank(self, query:str, docs:List[str]):
        self.eval()
        lenth = len(docs)
        querys = [query for i in range(lenth)]
        output = self.tokenizer(
            querys,
            docs,
            truncation = True,
            max_length = 512,
            padding = True,
            return_tensors = "pt"
        )
        self.to(self.device)
        output.to(self.device)
        scores_collection = []
        with torch.no_grad():
            scores = self.model(**output, return_dict=True).logits.view(-1,).float()
            scores = torch.sigmoid(scores)
            scores_collection.extend(scores.cpu().numpy().tolist())
        for i in range(lenth):
            print(query + " and " + docs[i], ",   relevance:", scores_collection[i])
        return scores_collection
    
    def recall(self, query, docs :list, k = 3):
        inputs =self.tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().cpu().numpy()
        top_indices = scores.argsort()[-k:][::-1]
        top_docs = [{"text": docs[int(i)], "score": scores[i]} for i in top_indices]
        return top_docs