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

    def complete(self, prompt, **kwargs):
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )

        return dspy.Prediction(text=response.text)

    def __call__(self, prompt, **kwargs):
        # Delegates to complete() — this is what DSPy internally expects
        return self.complete(prompt, **kwargs)

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
