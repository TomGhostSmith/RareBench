import os
import json
import time
import torch
import psutil
import requests
import subprocess
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration

class ModelServer:
    def __init__(self, command, cwd, env=None):
        if (env is None):
            env = os.environ
        self.env = env
        self.command = command
        self.cwd = cwd
        self.process = None
    
    def start(self, port):
        self.command += f" --port {port}"
        self.process = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, shell=True)
        ready = False
        URL = f"http://localhost:{port}/ready"
        timeout = 300
        startTime = time.time()
        while time.time() - startTime < timeout:
            try:
                response = requests.get(URL, timeout=2)
                if (response.status_code == 200):
                    ready = True
                    break
            except requests.exceptions.RequestException as e:
                pass
                
            time.sleep(2)
        
        if (not ready):
            raise Exception(f"failed to start the model in {timeout} seconds")
    
    def terminate(self):
        if self.process is not None:
            print("terminate server")
            parent = psutil.Process(self.process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            self.process.terminate()
            self.process.wait()
            self.process = None
        else:
            print("Nothing to terminate")
        

class Local_llm_handler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.modelServer:ModelServer = None

        if (self.model_name == 'glm-4-9b'):
            self.modelServer = ModelServer("conda run -n glm4 python /Software/Sland/GLM-4/server.py", cwd="/Software/Sland/GLM-4")
        elif (self.model_name == 'baichuan-2-13b'):
            self.modelServer = ModelServer("conda run -n base python /Software/Baichuan2/server.py", cwd="/Software/Baichuan2")
            
        
        self.modelServer.start(port=11000)
        print(f"{model_name} server started")


        # with open("mapping/local_llm_path.json", "r") as f:
        #     local_llm_path = json.load(f)
        # self.model_path = local_llm_path[model_name]
        # # return None
        
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if self.model_name == "chatglm3-6b":
        #     self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, device='cuda')
        # elif self.model_name in ["mistral-7b", "llama2-7b", "llama2-13b", "llama2-70b", "biomistral-7b", "medalpaca-7b"]:
        #     self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
        #     print(f"Using device: {self.device}")
        # elif self.model_name in ["huatuogpt2-7b"]:
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        #     self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        #     self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)

        # elif self.model_name in ["clinical-T5"]:
        #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_path, from_flax=True)
        # elif self.model_name in ["glm-4-9b"]:
        #     self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, attn_implementation="eager", device_map="auto")
        # elif self.model_name in ["baichuan-2-13b"]:
        #     return
        # self.model.eval()

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            # if self.model_name == "chatglm3-6b":
            #     result, history = self.model.chat(self.tokenizer, system_prompt + prompt, history=[])
            # elif self.model_name in ["llama2-7b", "llama2-13b", "llama2-70b", "medalpaca-7b"]:
            #     inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
            #     inputs = inputs.to(self.device)
            #     self.model.to(self.device)
            #     generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=1000)
            #     result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #     result = result.replace(system_prompt + prompt, "")
            # elif self.model_name in ["mistral-7b", "biomistral-7b"]:
            #     messages = [
            #         {"role": "user", "content": system_prompt + prompt}
            #     ]
            #     model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
            #     model_inputs = model_inputs.to(self.device)
            #     self.model.to(self.device)

            #     generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            #     result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #     result = result.replace(system_prompt + prompt, "")
            #     result = result.replace("[INST]", "")
            #     result = result.replace("[/INST]", "")
            # elif self.model_name in ["huatuogpt2-7b"]:
            #     self.model.to(self.device)
            #     messages = [
            #         {"role": "user", "content": system_prompt + prompt}
            #     ]
            #     result = self.model.HuatuoChat(self.tokenizer, messages)
                
            # elif self.model_name in ["clinical-T5"]:
            #     inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
            #     inputs = inputs.to(self.device)
            #     self.model.to(self.device)
            #     generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=4096)
            #     result = self.tokenizer.decode(generate_ids[0])
            #     print(result)

            # elif self.model_name in ["glm-4-9b"]:
            #     messages = [
            #         {"role": "user", "content": system_prompt + prompt}
            #     ]
            #     model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
            #     model_inputs = model_inputs.to(self.model.device)
            #     # self.model.to(self.device)

            #     generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            #     result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # result = result.replace(system_prompt + prompt, "")
            #     result = result.replace("[INST]", "")
            #     result = result.replace("[/INST]", "")
            if self.model_name in ["baichuan-2-13b", "glm-4-9b"]:
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                URL = "http://localhost:11000/chat"
                try:
                    response = requests.post(URL, json=messages, timeout=60)
                    if (response.status_code == 200):
                        result = response.json().get("response")
                        result = result.replace(system_prompt + prompt, "")
                    else:
                        print(f"Error: {response.status_code}, {response.text}")
                except requests.exceptions.RequestException as e:
                    print("Request failed: ", e)
                
            # print(f'Local LLM {self.model_name} time: {time.time() - t}')
            return result
        except Exception as e:
            print(e)
            return None

    def close(self):
        print("close handler")
        self.modelServer.terminate()