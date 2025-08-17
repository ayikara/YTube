import dspy
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from typing import Optional, List

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

class GeminiVertex(dspy.LM):
    def __init__(self, model_name="gemini-1.5-pro", temperature=0.7, max_tokens=2048):
        # DSPy requires this model name passed to super
        super().__init__(model=model_name)
        self.model = GenerativeModel(model_name)
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )

        return response.text  # Not wrapped in dspy.Prediction

# Set Gemini as the LM in DSPy
dspy.settings.configure(lm=GeminiVertex())
