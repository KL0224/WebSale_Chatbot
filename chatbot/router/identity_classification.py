import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from inspect import cleandoc

class IdentityClassificationLLM:
    def __init__(self, model_name="Qwen/Qwen1.5-1.8B-Chat"):
        self.device = "cpu"
        print(f"Loading model {model_name} to {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        ).to(self.device)

        print("Model đã load thành công.")

    def classify(self, query):
        # Sử dụng cleandoc để prompt sạch sẽ, không bị dính khoảng trắng lùi lề của Python
        from inspect import cleandoc

        prompt = cleandoc(f"""
        You are an intent classification system for an e-commerce assistant.

        TASK:
        Classify the user message into EXACTLY ONE label.

        LABELS:
        - policy
        - product
        - chitchat

        --------------------------------
        LABEL DEFINITIONS:

        policy:
        Questions about:
        - warranty
        - return / refund
        - shipping / delivery
        - payment methods
        - store rules
        - order processing
        - invoices, bills
        - policies and procedures

        product:
        Questions about:
        - price
        - features
        - specifications
        - availability
        - comparison
        - recommendations
        - product quality
        - usage suitability

        chitchat:
        - greetings
        - small talk
        - emotions
        - compliments
        - casual conversation
        - unrelated topics
        --------------------------------

        STRICT RULES:
        - Return ONLY ONE word
        - No explanation
        - No punctuation
        - No extra text
        - No formatting
        - No JSON
        - No markdown

        --------------------------------
        EXAMPLES:

        User: "Shop có chính sách bảo hành như thế nào?"
        Label: policy

        User: "iPhone 15 Pro Max giá bao nhiêu?"
        Label: product

        User: "Chào bạn, hôm nay bạn thế nào?"
        Label: chitchat

        --------------------------------
        INPUT:
        User: "{query}"

        OUTPUT:
        Label:
        """)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Cấu hình sinh từ (Generation)
        gen_config = GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.model.generation_config = gen_config

        with torch.no_grad():
            output = self.model.generate(
                **inputs
            )

        # CHỈ decode phần token mới (loại bỏ phần prompt gốc)
        new_tokens = output[0][input_length:]
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # Chuẩn hóa đầu ra (Normalization)
        if "policy" in prediction:
            return "policy"
        elif "product" in prediction:
            return "product"
        else:
            return "chitchat"


# --- Cách sử dụng ---
if __name__ == "__main__":
    # Thay tên model bạn đang dùng vào đây
    classifier = IdentityClassificationLLM("Qwen/Qwen2.5-1.5B-Instruct")
    queries = ["Bạn ơi, hôm nay bạn khỏe không", "Chính sách giao hàng của cửa hàng thế nào?", "Điện thoại SamSung còn màu đỏ không?"]

    for test_query in queries:
        result = classifier.classify(test_query)

        print(f"\nQuery: {test_query}")
        print(f"Final Label: {result}")