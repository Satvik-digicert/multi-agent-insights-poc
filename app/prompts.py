### ---- Define Tables and Relationships ---- ###

TABLES_INFO = """
We have the following tables in our Databricks environment under the schema `lake_prod.dcone_enterprise_gp2`:

TABLES PROVIDED:
- `certificate` (8.95M rows)
- `enrollment` (5.94M rows)
- `profile` (3.7K rows)
- `business_unit` (4.4K rows)
- `automation_inventory` (236K rows)
- `account` (8.3K rows)

Key relationships:
- `certificate.enrollment_id` → `enrollment.id`
- `enrollment.profile_id` → `profile.id`
- `profile.business_unit_id` → `business_unit.id`
- `profile.account_id` → `account.id`
- `automation_inventory.account_id` → `account.id`

SCHEMA OVERVIEW:
certificate:
- id, status, enrollment_id, valid_from, valid_to, profile_id, account_id, business_unit_id
- signing_algorithm, key_length, public_key_algorithm
- automation_instance_count, discovery_instance_count, kafka_timestamp_utc

enrollment:
- id, status, email, profile_id, seat_id, created_at, expires_at

profile:
- id, name, account_id, business_unit_id, enrollment_method_id, authentication_method_id
- status, created_at, updated_at

business_unit:
- id, name, account_id, created_at

account:
- id, name, start_date, end_date, active

automation_inventory:
- id, account_id, automation_status, profile_id, last_enrollment_id, created_at, updated_at
"""

QUERY_AGENT_PROMPT = f"""
    You are a BI assistant. Based on the following table schemas and relationships, generate 5 insightful SQL queries
    that would provide meaningful insights to a business user.
    
    You are a certificate lifecycle insights analyst. Analyze the provided relational schema for core tables involved in Certificate Lifecycle Management (CLM). 
    Use this information to generate insightful SQL queries that would provide meaningful insights to a business user.

    {TABLES_INFO}

    GOAL:
    Generate queries on this data that improve the insights :
    - Certificate lifecycle visibility
    - Operational efficiency
    - Security compliance
    - PKI hygiene and automation coverage
    
    INSIGHTS CAN BE CATEGORIZED AS:
    1. Trend-Based Insights — Use month-on-month patterns, adoption curves, revocation/issuance surges.
    2. Non-Trend Insights — Point-in-time anomalies, misconfigurations, or risky clusters.
    
    EXAMPLES OF INSIGHTS TO GENERATE:
    - Certificate Expiration Risk: Certs expiring in 30/60/90 days, % renewed late
    - Legacy Validation Methods: % using WHOIS/Email vs DNS/HTTPS
    - TLS Version Compliance: % using TLS 1.2 or older
    - Automation Gaps: % public certs not automated (from automation_inventory)
    - EKU Misuse: Certs with both server + client auth (blocked by Chrome May 2026)
    - Expired Certificates Spike: Month-on-month increase in expired certs
    - High Revocation Rates: Short-lived certs revoked frequently
    - Self-Signed Certs: % not issued by trusted CA
    - Key Length Compliance: Certs with weak keys like RSA 1024
    - Wildcard Usage Surge: Trend in wildcard certs
    - Known Vulnerabilities: Weak algorithms or metadata exposure
    - Ownership Gaps: Certs with null owner links (profile/account)
    - Unmapped Certificates: No linked assets or domains
    - Operational Risk Clusters: Certs expiring in the same 15–30 day window
    
    ALSO FLAG:
    - Missing critical fields (`valid_to`, `profile_id`, etc.)
    - Business units/accounts with high revocation or failure rates
    - Certs nearing expiry without automation enabled
    
    RESPONSE FORMAT:
    Provide the queries in raw SQL format that are separated by delimiter '#####'  and do not add any other content in the responses apart from the saw sql queries  .    
    """



INSIGHTS_PROMPT = f"""
        
        {TABLES_INFO}
        
        You are a certificate lifecycle insights analyst. Given the following query and its results from tables of Certificate Lifecycle Management (CLM),  
        Use this information to extract actionable insights.
        
        GOAL:
        Extract insights from this data that improve:
        - Certificate lifecycle visibility
        - Operational efficiency
        - Security compliance
        - PKI hygiene and automation coverage
        
        CATEGORIZE INSIGHTS AS:
        1. Trend-Based Insights — Use month-on-month patterns, adoption curves, revocation/issuance surges.
        2. Non-Trend Insights — Point-in-time anomalies, misconfigurations, or risky clusters.
        
        EXAMPLES OF INSIGHTS TO GENERATE:
        - Certificate Expiration Risk: Certs expiring in 30/60/90 days, % renewed late
        - Legacy Validation Methods: % using WHOIS/Email vs DNS/HTTPS
        - TLS Version Compliance: % using TLS 1.2 or older
        - Automation Gaps: % public certs not automated (from automation_inventory)
        - EKU Misuse: Certs with both server + client auth (blocked by Chrome May 2026)
        - Expired Certificates Spike: Month-on-month increase in expired certs
        - High Revocation Rates: Short-lived certs revoked frequently
        - Self-Signed Certs: % not issued by trusted CA
        - Key Length Compliance: Certs with weak keys like RSA 1024
        - Wildcard Usage Surge: Trend in wildcard certs
        - Known Vulnerabilities: Weak algorithms or metadata exposure
        - Ownership Gaps: Certs with null owner links (profile/account)
        - Unmapped Certificates: No linked assets or domains
        - Operational Risk Clusters: Certs expiring in the same 15–30 day window
        
        ALSO FLAG:
        - Missing critical fields (`valid_to`, `profile_id`, etc.)
        - Business units/accounts with high revocation or failure rates
        - Certs nearing expiry without automation enabled
        
        RESPONSE FORMAT:
        Group insights into:
        - Trend-Based Insights
        - Non-Trend-Based Insights
        - Optimization Recommendations
        
        Each insight must include:
        - Insight Summary
        - Supporting Evidence (stat/metric)
        - Recommended Action
        - Score 0 to 1 indicating how insightful the data is
        
        The goal is to produce a dashboard-ready intelligence summary. Be concise, analytical, and practical.

"""

