import dspy
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

class GeminiVertex(dspy.BaseLM):
    def __init__(self, model_name="gemini-1.5-pro", max_tokens=2048, temperature=0.7):
        super().__init__(model=model_name)
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def __call__(self, prompt, **kwargs):
        """Main method called by DSPy"""
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return [response.text]
        except Exception as e:
            print(f"Error generating content: {e}")
            return [""]

    def generate(self, prompt, **kwargs):
        """Alternative method that some DSPy versions expect"""
        return self.__call__(prompt, **kwargs)

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
