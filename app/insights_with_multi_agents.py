# multi_agent_insights.py

import os
from dotenv import load_dotenv  # Add this import
import json
import pandas as pd
from typing import TypedDict, Annotated, List
from databricks import sql
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown


### ---- Configuration ---- ###
load_dotenv()  # Load environment variables from .env file

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST')
DATABRICKS_HTTP_PATH = os.environ.get('DATABRICKS_HTTP_PATH')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

TEMPERATURE= os.environ.get('TEMPERATURE')
AZURE_DEPLOYMENT= os.environ.get('AZURE_DEPLOYMENT')
AZURE_ENDPOINT= os.environ.get('AZURE_ENDPOINT')
AZURE_API_VERSION= os.environ.get('AZURE_API_VERSION')


DATABRICKS_CONFIG = {
    "server_hostname": DATABRICKS_HOST,
    "http_path": DATABRICKS_HTTP_PATH,
    "access_token":  DATABRICKS_TOKEN
}

### ---- Define Tables and Relationships ---- ###

TABLES_INFO = """
We have the following tables in our Databricks environment:

1. lake_prod.dcone_enterprise_gp2.tag (id, name, created_at, status, account_id, kafka_timestamp_utc)
2. lake_prod.dcone_enterprise_gp2.profile_tag_mapping (profile_id, tag_id, kafka_timestamp_utc)
3. lake_prod.dcone_enterprise_gp2.profile (id, name, account_id, business_unit_id, template_id, ica_id, authentication_method_id, enrollment_method_id, days_till_enrollment_expires, fields, active, suspended, notify_requestor, additional_recipients, cert_delivery_format, ca_chain_value, renewal_window_days, automated_renewal, embed_enrollment_code_in_url, global_enrollment_code, allow_validity_from_rest_api, duplicate_certificate_allowed, auth_fields, seat_id_mapping, enrollment_code_length, enable_dual_admin_approval, ddc_settings, scep_settings, msae_settings, key_escrow_policy, intune_app_config_id, enable_ldap_search, fixed_enrollment_url_enabled, ica_name, cc_settings, dta_settings, action_needed_reason, updated_by, created_by, ca_settings, api_token_ids, connector_id, max_number_of_bad_attempts, smime_validation_type, smime_cert_profile_type, grace_period, last_validated_at, status, allowed_ip_fqdn, custom_extensions_config, ssp_portal, dns_integration_id, ios_settings, est_settings, custom_extended_key_usages, ssp_auth_operations, ssp_open_operations, created_at, updated_at, description, allowed_user_group, intune_connector, additional_delivery_formats, kafka_timestamp_utc)

Relations:
- lake_prod.dcone_enterprise_gp2.tag.id → lake_prod.dcone_enterprise_gp2.profile_tag_mapping.tag_id
- lake_prod.dcone_enterprise_gp2.profile_tag_mapping.profile_id → lake_prod.dcone_enterprise_gp2.lake_prod.profile.id
"""

### ---- Graph State ---- ###
class GraphState(TypedDict):
    queries_executed: Annotated[List[str], "List of executed SQL queries"]
    query_results: Annotated[List[dict], "List of dicts: {'query': str, 'result': DataFrame}"]
    insights: Annotated[List[dict], "List of dicts: {'query': str, 'insight': str, 'score': float}"]

### ---- LLM Setup ---- ###

llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    temperature=TEMPERATURE
)
# Setup rich consoles for HTML and Markdown logs
console = Console(record=True)
html_log_path = "insights.html"  # changed filename
md_log_path = "insights.md"      # changed filename
persisted_html_log_path = "insights_persisted.html"
persisted_md_log_path = "insights_persisted.md"

def persist_rich_logs():
    # Export HTML log for this run (overwrite)
    html = console.export_html(clear=False)
    with open(html_log_path, "w") as f:
        f.write(html)
    # Export Markdown log for this run (overwrite)
    md = console.export_text(clear=False)
    with open(md_log_path, "w") as f:
        f.write(md)
    # Append HTML log to persisted file
    with open(persisted_html_log_path, "a") as f:
        f.write(html)
        f.write("\n<!-- --- End of Run --- -->\n")
    # Append Markdown log to persisted file
    with open(persisted_md_log_path, "a") as f:
        f.write(md)
        f.write("\n--- End of Run ---\n")

### ---- Databricks Query ---- ###
def execute_query(query: str) -> pd.DataFrame:
    with sql.connect(
        server_hostname=DATABRICKS_CONFIG["server_hostname"],
        http_path=DATABRICKS_CONFIG["http_path"],
        access_token=DATABRICKS_CONFIG["access_token"],
        _tls_no_verify=True
    ) as connection:
        cursor = connection.cursor()
        console.log(f"[QUERY LOG] Executing SQL Query: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

### ---- Agent: Query Generator ---- ###
def query_agent(_: dict) -> GraphState:
    prompt = f"""
    You are a BI assistant. Based on the following table schemas and relationships, generate 3 insightful SQL queries
    that would provide meaningful insights to a business user.

    {TABLES_INFO}

    Provide the queries in raw SQL format that are separated by delimiter '#####'  and do not add any other content in the responses apart from the saw sql queries  .
    """
    response = llm.invoke(prompt)
    console.rule("[bold blue]=== LLM Query Generation Response ===")
    queries = response.content.strip().split("#####")
    print(f' queries : {queries}')

    results = []
    executed_queries = []
    for query in queries:
        try:
            print(f' query : {query}')
            df = execute_query(query)
            print(f'df shape: {df.shape} ')
            results.append({"query": query, "result": df})
            executed_queries.append(query)
        except Exception as e:
            console.log(f"[QUERY ERROR] Query failed: {query}\nError: {str(e)}")
            continue

    return {
        "queries_executed": executed_queries,
        "query_results": results,
        "insights": []
    }

### ---- Agent: Insight Generator ---- ###
def insight_generator(state: GraphState) -> GraphState:
    insights = []
    for item in state["query_results"]:
        query = item["query"]
        df = item["result"]
        prompt = f"""
        You are a data analyst. Given the following query and its results, generate a short insight (2-3 sentences).
        Also assign a score from 0 to 1 indicating how insightful the data is.

        SQL Query:
        {query}

        Result:
        {df.to_json(orient='records')[:4000]}

        Format:
        Insight: <your insight>
        Score: <score>
        """
        print(f' insight : {query}')
        print(f' insight : {df.head(5)} ')

        result = llm.invoke(prompt)
        # Replace the parsing block in insight_generator with this:
        try:
            print(f' insight result : {result.content}')
            # Split and filter out empty lines
            lines = [line for line in result.content.splitlines() if line.strip()]
            insight = next((line.replace("Insight:", "").strip() for line in lines if line.startswith("Insight:")), "")
            score_line = next((line for line in lines if line.startswith("Score:")), "Score: 0")
            score = float(score_line.replace("Score:", "").strip())
            insights.append({"query": query, "insight": insight, "score": score})
            console.rule("[bold green]=== Insight Generated ===")
            console.print(Markdown(f"**Insight:** {insight}\n\n**Score:** {score}\n\n**Query:**\n{query}"))
        except Exception as e:
            console.log(f"[INSIGHT ERROR] Failed to parse insight for query: {query}\nError: {str(e)}")
            continue

    return {
        **state,
        "insights": sorted(insights, key=lambda x: x["score"], reverse=True)
    }

### ---- Graph Wiring ---- ###
graph = StateGraph(GraphState)
graph.add_node("query_agent", query_agent)
graph.add_node("insight_agent", insight_generator)

graph.set_entry_point("query_agent")
graph.add_edge("query_agent", "insight_agent")
graph.add_edge("insight_agent", END)

multi_agent_graph = graph.compile()

### ---- Execute ---- ###
if __name__ == "__main__":
    console.rule("[bold magenta]=== Executing Multi-Agent BI Workflow ===")
    result = multi_agent_graph.invoke({})

    console.rule("[bold blue]=== Executed Queries ===")
    for q in result["queries_executed"]:
        console.print(Markdown(f"- `{q}`"))

    console.rule("[bold green]=== Insights (Sorted by Score) ===")
    for item in result["insights"]:
        console.print(Markdown(
            f"\n**Score:** {item['score']:.2f}\n\n**Insight:** {item['insight']}\n\n**Query:**\n{item['query']}\n"
        ))

    console.rule("[bold yellow]=== Graph Visualization (Mermaid PNG) ===")
    png_data = multi_agent_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    console.print("Graph saved as `graph.png`")

    persist_rich_logs()