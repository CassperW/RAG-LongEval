from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

RAG_PROMPT = """
[System Introduction]
You are an AI assistant that generates concise answers based on the provided context. Below are three relevant excerpts retrieved from a knowledge base. Use these excerpts to generate a short and accurate response to the user's question.

[Context Section]
{context}

[User Question]
Question: {user_question}

[Instruction to LLM]
Provide the most concise answer possible based on the context, following these rules:
1. Use the minimum number of words necessary to accurately answer the question, do not use full sentences, explanations, or any unnecessary words.
2. Do not include any additional context or clarification in your answer.
3. For questions about a specific entity (person, company, team, and so on), provide only the full name.

Examples:
Question: What is the capital of France? Answer: Paris
Question: Is the sky blue? Answer: Yes
Question: Who is the CEO of Tesla? Answer: Elon Musk

Answer:
"""

LONG_CONTEXT_PROMPT = """
[System Introduction]
You are an AI assistant that generates concise answers based on a long document. Below is a comprehensive text that contains information related to the user's question. Read the text carefully and generate a short and accurate response to the user's question.

[Document Section]
{long_document}

[User Question]
Question: {user_question}

[Instruction to LLM]
Provide the most concise answer possible based on the context, following these rules:
1. Use the minimum number of words necessary to accurately answer the question, do not use full sentences, explanations, or any unnecessary words.
2. Do not include any additional context or clarification in your answer.
3. For questions about a specific entity (person, company, team, and so on), provide only the full name.

Examples:
Question: What is the capital of France? Answer: Paris
Question: Is the sky blue? Answer: Yes
Question: Who is the CEO of Tesla? Answer: Elon Musk

Answer:
"""

mistral_models_path = '/home/yujie/models/mistral7b'

class Mistral():
    def __init__(self, path = mistral_models_path):
        self.tokenizer = MistralTokenizer.from_file(f"{path}/tokenizer.model.v3")
        self.model = Transformer.from_folder(path)

    def generate(self, question, context):
        # prompt = RAG_PROMPT.format(context=context, user_question=question)
        prompt = LONG_CONTEXT_PROMPT.format(long_document=context, user_question=question)
    
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], self.model, max_tokens=32, temperature=0.0, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return result