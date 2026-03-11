"""Medical diagnostic agents using LangChain."""

import asyncio
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class MedicalAgent:
    """Base class for medical diagnostic agents."""

    def __init__(self, medical_report: str, role: str, model_name: str = "gpt-4o"):
        """Initialize medical agent with report, role, and model."""
        self.medical_report = medical_report
        self.role = role

        # Get API key from environment (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API key with OpenAI client
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        else:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable.")

        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create role-specific prompt template."""
        templates = {
            "Cardiologist": """
                Act like a cardiologist. You will receive a medical report of a patient.
                Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
                Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient's symptoms. Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities, that might be missed on routine testing.
                Recommendation: Provide guidance on any further cardiac testing or monitoring needed to ensure there are no hidden heart-related concerns. Suggest potential management strategies if a cardiac issue is identified.
                Please only return the possible causes of the patient's symptoms and the recommended next steps.

                Medical Report: {medical_report}
            """,
            "Psychologist": """
                Act like a psychologist. You will receive a patient's report.
                Task: Review the patient's report and provide a psychological assessment.
                Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.
                Recommendation: Offer guidance on how to address these mental health concerns, including therapy, counseling, or other interventions.
                Please only return the possible mental health issues and the recommended next steps.

                Patient's Report: {medical_report}
            """,
            "Pulmonologist": """
                Act like a pulmonologist. You will receive a patient's report.
                Task: Review the patient's report and provide a pulmonary assessment.
                Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.
                Recommendation: Offer guidance on how to address these respiratory concerns, including pulmonary function tests, imaging studies, or other interventions.
                Please only return the possible respiratory issues and the recommended next steps.

                Patient's Report: {medical_report}
            """,
        }

        return PromptTemplate.from_template(templates[self.role])

    async def run(self) -> str:
        """Run the agent analysis."""
        print(f"{self.role} is running...")
        try:
            response = await self.chain.ainvoke({"medical_report": self.medical_report})
            return response
        except Exception as e:
            print(f"Error occurred in {self.role}: {e}")
            return f"Error: {e!s}"


class MultidisciplinaryTeam:
    """Agent that synthesizes reports from multiple specialists."""

    def __init__(
        self, cardiologist_report: str, psychologist_report: str, pulmonologist_report: str, model_name: str = "gpt-4o"
    ):
        """Initialize multidisciplinary team with specialist reports."""
        self.cardiologist_report = cardiologist_report
        self.psychologist_report = psychologist_report
        self.pulmonologist_report = pulmonologist_report
        self.model_name = model_name

        # Create synthesis prompt
        synthesis_prompt = f"""
            Act like a multidisciplinary team of healthcare professionals.
            You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.
            Task: Review the patient's medical report from the Cardiologist, Psychologist, and Pulmonologist, analyze them and come up with a list of 3 possible health issues of the patient.
            Just return a list of bullet points of 3 possible health issues of the patient and for each issue provide a reason.

            Cardiologist Report: {cardiologist_report}

            Psychologist Report: {psychologist_report}

            Pulmonologist Report: {pulmonologist_report}
        """

        # Create the model using the same pattern as specialist agents
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        # Get API key from environment (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        else:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable.")

        # Create synthesis chain
        self.chain = PromptTemplate.from_template(synthesis_prompt) | model | StrOutputParser()

    async def run(self) -> str:
        """Run the multidisciplinary analysis."""
        print("MultidisciplinaryTeam is running...")
        try:
            response = await self.chain.ainvoke({
                "cardiologist_report": self.cardiologist_report,
                "psychologist_report": self.psychologist_report,
                "pulmonologist_report": self.pulmonologist_report,
            })
            return response
        except Exception as e:
            print(f"Error occurred in MultidisciplinaryTeam: {e}")
            return f"Error: {e!s}"


class Cardiologist(MedicalAgent):
    """Cardiologist specialist agent."""

    def __init__(self, medical_report: str, model_name: str = "gpt-4o"):
        """Initialize cardiologist agent with medical report and model."""
        super().__init__(medical_report, "Cardiologist", model_name)


class Psychologist(MedicalAgent):
    """Psychologist specialist agent."""

    def __init__(self, medical_report: str, model_name: str = "gpt-4o"):
        """Initialize psychologist agent with medical report and model."""
        super().__init__(medical_report, "Psychologist", model_name)


class Pulmonologist(MedicalAgent):
    """Pulmonologist specialist agent."""

    def __init__(self, medical_report: str, model_name: str = "gpt-4o"):
        """Initialize pulmonologist agent with medical report and model."""
        super().__init__(medical_report, "Pulmonologist", model_name)


async def run_medical_diagnosis(medical_report: str, model_name: str = "gpt-4o") -> str:
    """Run the complete medical diagnosis pipeline.

    Args:
        medical_report: The patient's medical report text
        model_name: The name of the model to use

    Returns:
        Final diagnosis from the multidisciplinary team

    """
    # Create specialist agents with their own model instances
    agents = {
        "Cardiologist": Cardiologist(medical_report, model_name),
        "Psychologist": Psychologist(medical_report, model_name),
        "Pulmonologist": Pulmonologist(medical_report, model_name),
    }

    # Run all agents concurrently
    tasks = [agent.run() for agent in agents.values()]
    responses = await asyncio.gather(*tasks)

    # Map responses back to agent names
    agent_names = list(agents.keys())
    specialist_reports = dict(zip(agent_names, responses, strict=False))

    # Create and run multidisciplinary team
    team = MultidisciplinaryTeam(
        cardiologist_report=specialist_reports["Cardiologist"],
        psychologist_report=specialist_reports["Psychologist"],
        pulmonologist_report=specialist_reports["Pulmonologist"],
        model_name=model_name,
    )

    # Run the team and get final diagnosis
    final_diagnosis = await team.run()
    return final_diagnosis
