import os
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

def get_chat_response(question: str, dataset_context: str) -> str:
    """
    Uses Anthropic Claude to answer questions about the uploaded dataset.
    Falls back to rule-based answers if API key is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    print("api key", api_key)

    if not api_key:
        return _fallback_response(question, dataset_context)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = f"""You are an expert Data Scientist AI assistant embedded in a machine learning platform.
The user has uploaded a dataset and you have FULL knowledge of it from the context below.

Your job:
- Answer questions clearly and accurately based ONLY on the dataset context provided.
- Use specific numbers, column names, and statistics from the dataset.
- Format answers with bullet points and bold text where helpful.
- Keep answers concise but informative (2-5 sentences max unless asked for more).
- If asked about something not in the dataset, say so clearly.

DATASET CONTEXT:
{dataset_context}
"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": question}]
        )

        return message.content[0].text

    except Exception as e:
        return f"⚠️ Chatbot error: {str(e)}"


def _fallback_response(question: str, context: str) -> str:
    """Simple keyword-based fallback when no API key is set."""
    q = question.lower()
    lines = context.split('\n')

    def find_in_context(keyword):
        for line in lines:
            if keyword.lower() in line.lower():
                return line.strip()
        return None

    if any(w in q for w in ['how many', 'rows', 'shape', 'size']):
        line = find_in_context('Shape') or find_in_context('rows')
        return f"📊 {line}" if line else "Dataset shape info is in the context above."

    if any(w in q for w in ['missing', 'null', 'nan']):
        line = find_in_context('Missing')
        return f"⚠️ {line}" if line else "No missing value info found."

    if any(w in q for w in ['accuracy', 'score', 'performance', 'model']):
        line = find_in_context('Metrics') or find_in_context('accuracy') or find_in_context('r2')
        return f"🤖 {line}" if line else "Model performance info is in the ML Results tab."

    if any(w in q for w in ['feature', 'important', 'column']):
        line = find_in_context('Feature Importance')
        return f"🔍 {line}" if line else "Check the Feature Importance section in ML Results."

    if any(w in q for w in ['target', 'predict']):
        line = find_in_context('Target')
        return f"🎯 {line}" if line else "Target column info is in the dataset context."

    return ("💬 I can answer questions about your dataset — try asking about:\n"
            "• Missing values\n• Model accuracy\n• Feature importance\n• Dataset shape\n"
            "• Column statistics\n\n"
            "*(Set ANTHROPIC_API_KEY for full AI-powered responses)*")


def build_dataset_context(df, target_col, task_type, result) -> str:
    """Build rich context string for the chatbot."""
    try:
        describe_str = df.describe().round(3).to_string()
    except:
        describe_str = "N/A"

    try:
        if task_type == 'classification':
            target_info = df[target_col].value_counts().head(10).to_string()
        else:
            target_info = str(df[target_col].describe().round(3))
    except:
        target_info = "N/A"

    context = f"""
=== DATASET OVERVIEW ===
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Task Type: {task_type.capitalize()}
Target Column: {target_col}
Feature Columns: {result['feature_cols']}

=== COLUMN TYPES ===
Numeric: {df.select_dtypes(include='number').columns.tolist()}
Categorical: {df.select_dtypes(include=['object','category']).columns.tolist()}

=== MISSING VALUES ===
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().any() else "No missing values"}

=== STATISTICAL SUMMARY ===
{describe_str}

=== TARGET DISTRIBUTION ===
{target_info}

=== ML MODEL RESULTS ===
Best Model: {result['best_model_name']}
Metrics: {result['metrics']}

=== TOP FEATURE IMPORTANCE ===
{result.get('feature_importance', {})}
"""
    return context

