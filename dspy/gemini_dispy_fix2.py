import dspy
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

class GeminiVertex(dspy.BaseLM):
    def __init__(self, model_name="gemini-1.5-pro", max_tokens=2048, temperature=0.7, **kwargs):
        super().__init__(model=model_name, **kwargs)
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def basic_request(self, prompt, **kwargs):
        """Core method that DSPy calls internally"""
        if isinstance(prompt, list):
            prompt = "\n".join(str(p) for p in prompt)
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """DSPy calls this with specific parameters"""
        response_text = self.basic_request(prompt, **kwargs)
        
        # DSPy expects a list of completions
        completions = [response_text] if response_text else [""]
        
        if return_sorted:
            return completions
        
        return completions

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
