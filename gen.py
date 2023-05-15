# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import BertTokenizer, BertForMaskedLM, BertLMHeadModel
# from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertLMHeadModel
# import torch
# import numpy as np

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load your embeddings
# embeddings = np.load("activity/0a09f8fc-ff87-4210-b682-d2ae38af33eb.npy")
# print("shape of embeddings: ", embeddings.shape)

# encodings = torch.from_numpy(embeddings)  # assuming 1 batch of encodings

# # # load the tokenizer and model
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # # model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
# # model = BertLMHeadModel.from_pretrained('bert-base-uncased').to(device)

# # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# # model = AlbertLMHeadModel.from_pretrained('albert-base-v2').to(device)
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertForMaskedLM.from_pretrained('albert-base-v2').to(device)


# # tokenizer = T5Tokenizer.from_pretrained('t5-small', )
# # model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)


# # Load the GPT-2 tokenizer and model
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# # Convert the encoding to a tensor and feed it to the model
# input_ids = torch.tensor(encodings, dtype=torch.long)
# with torch.no_grad():
#     output = model.generate(
#         input_ids.to(device),
#         do_sample=True,
#         max_length=512,  # set the maximum length of the generated text
#         top_p=0.95,  # set the top-p sampling parameter
#         # set the top-k sampling parameter (disable it by setting to 0)
#         top_k=0
#     )

# # Decode the generated text using the tokenizer
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import BertTokenizer, BertForMaskedLM, BertLMHeadModel
# from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertLMHeadModel
# import torch
# import numpy as np

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load your embeddings
# embeddings = np.load("activity/0a09f8fc-ff87-4210-b682-d2ae38af33eb.npy")
# print("shape of embeddings: ", embeddings.shape)

# encodings = torch.from_numpy(embeddings)  # assuming 1 batch of encodings

# # # load the tokenizer and model
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # # model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
# # model = BertLMHeadModel.from_pretrained('bert-base-uncased').to(device)

# # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# # model = AlbertLMHeadModel.from_pretrained('albert-base-v2').to(device)
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertForMaskedLM.from_pretrained('albert-base-v2').to(device)


# # tokenizer = T5Tokenizer.from_pretrained('t5-small', )
# # model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)


# # Load the GPT-2 tokenizer and model
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# # Convert the encoding to a tensor and feed it to the model
# input_ids = torch.tensor(encodings, dtype=torch.long)
# with torch.no_grad():
#     output = model.generate(
#         input_ids.to(device),
#         do_sample=True,
#         max_length=512,  # set the maximum length of the generated text
#         top_p=0.95,  # set the top-p sampling parameter
#         # set the top-k sampling parameter (disable it by setting to 0)
#         top_k=0
#     )

# # Decode the generated text using the tokenizer
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import torch.nn as nn

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your embeddings
embeddings = np.load("activity/0a09f8fc-ff87-4210-b682-d2ae38af33eb.npy")
print("shape of embeddings: ", embeddings.shape)

encodings = torch.from_numpy(embeddings)  # assuming 1 batch of encodings

tokenizer = T5Tokenizer.from_pretrained('t5-small', )
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)


# Load the GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Convert the encoding to a tensor and feed it to the model
input_ids = torch.tensor(encodings, dtype=torch.long)
input_ids = input_ids.to(device)
with torch.no_grad():
    # Split the input_ids tensor across two GPUs
    input_ids_1 = input_ids[:torch.cuda.device_count() // 2]
    input_ids_2 = input_ids[torch.cuda.device_count() // 2:]

    # Create two DataParallel models
    model_1 = nn.DataParallel(model, device_ids=[0])
    model_2 = nn.DataParallel(model, device_ids=[1])

    # Generate text using the two models
    output_1 = model_1.module.generate(
        input_ids_1.to(device),
        do_sample=True,
        max_length=512,  # set the maximum length of the generated text
        top_p=0.95,  # set the top-p sampling parameter
        # set the top-k sampling parameter (disable it by setting to 0)
        top_k=0
    )
    output_2 = model_2.generate(
        input_ids_2.to(device),
        do_sample=True,
        max_length=512,  # set the maximum length of the generated text
        top_p=0.95,  # set the top-p sampling parameter
        # set the top-k sampling parameter (disable it by setting to 0)
        top_k=0
    )

    # Concatenate the two output tensors
    output = torch.cat([output_1, output_2], dim=0)

    # Decode the generated text using the tokenizer
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)