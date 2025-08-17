import dspy
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

class GeminiVertex(dspy.LM):
    def __init__(self, model_name="gemini-1.5-pro", max_tokens=2048, temperature=0.7):
        super().__init__(model=model_name)
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def basic_request(self, prompt, **kwargs):
        """DSPy expects this method for basic completions"""
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )

        return [response.text]

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """DSPy calls this method with specific parameters"""
        response = self.basic_request(prompt, **kwargs)
        
        if return_sorted:
            return response
        
        return response[0] if response else ""

# Configure DSPy to use Gemini
dspy.settings.configure(lm=GeminiVertex())

class SimpleQA(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

predict = dspy.Predict(SimpleQA)

response = predict(
    context="Bananas are rich in potassium and fiber.",
    question="What nutrients are bananas rich in?"
)

print(response.answer)
