# Business Problem “Given a product requirement spec, generate PyTest-style unit test cases covering edge cases and invalid inputs.”
# Initial prompt "Given the function below and its description, write test cases using PyTest."


import dsp

class StructuredPromptGenerator(dsp.Module):
    def __init__(self):
        super().__init__()
        self.generator = dsp.Predict("structured_prompt")

    def forward(self, business_problem: str, initial_prompt: str) -> str:
        return self.generator(
            instruction="Rewrite the initial prompt into a complete, structured prompt aligned to the business problem.",
            business_problem=business_problem,
            initial_prompt=initial_prompt
        )


 trainset = [
    {
        "business_problem": "Generate test cases for REST API endpoints based on Swagger docs.",
        "initial_prompt": "Write PyTest functions for an API.",
        "structured_prompt": "You are given a Swagger file for a REST API. For each endpoint, generate PyTest unit tests that validate all parameters, expected status codes, and edge cases..."
    },
    {
        "business_problem": "Summarize customer complaints by sentiment.",
        "initial_prompt": "Summarize the complaints.",
        "structured_prompt": "You are a sentiment analysis agent. Given a list of customer complaint texts, group them into Positive, Neutral, or Negative, and summarize each group."
    }
]


from dsp.utils import simple_optimization

module = StructuredPromptGenerator()
optimized_module = simple_optimization(module, trainset)


business_problem = "Convert a marketing requirement into social media post text with call-to-action"
initial_prompt = "Generate a social post"

structured_prompt = optimized_module(
    business_problem=business_problem,
    initial_prompt=initial_prompt
)
print(structured_prompt)
