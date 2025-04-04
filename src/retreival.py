from typing import List
import numpy as np
from pydantic import BaseModel, Field

from  .models import llm


class SubQueries(BaseModel):
    """List consisting of Sub-Queries as string values."""
    sub_queries: List[str] = Field(description="List consisting of string values", max_length=2)


def multi_stage_retrieval(user_query, vector_store, llm):
    structured_llm = llm.with_structured_output(SubQueries)

    generate_subq_prompt = f"""Given the following user query, break it down into smaller, specific sub-queries that cover different aspects of the original query.
    Each sub-query should focus on a particular component or perspective of the main question. Return only a Python List which contains the generated sub-queries as string values.
    Strictly generate only 2 sub-queries and not more than 2.

    User Query: '{user_query}'
    Sub-Queries:"""

    messages=[{"role": "user", "content": generate_subq_prompt}]
    response: SubQueries = structured_llm.invoke(messages)

    context=""
    for i, query in enumerate(response.sub_queries):
        print(i)
        #!! IMPORTANT NOTE : Need to add bm25 search
        relevant_chunks = vector_store.similarity_search(query, k=3)
        #!! After getting all the chunks add a thershold which only considers chunks above the thershold similarity (so only relevant chunks are passed)

        context += f"\n\n###Sub-Query {i+1}: {query}\n\n" + "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk.page_content for i, chunk in enumerate(relevant_chunks)])

    return context

def retreival(vector_store, multi_stage=True, llm=llm):
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        if multi_stage:
            context = multi_stage_retrieval(user_query, vector_store, llm)
             
        # relevant_chunks = vector_store.similarity_search(user_query, k=3)
        # context += f"\n\n###User Query : {user_query}\n\n" + "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk.page_content for i, chunk in enumerate(relevant_chunks)])
        print(context)
        generate_prompt = f"""Please answer the following question based on the context provided.

        Before answering, analyze all documents corresponding to each sub-query in the context and identify if it contains the answer to the question.
        Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
        Also, make sure to list the most relevant documents first and then answer the question based on those documents only.

        If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
        '\n\nContext:\n{context}\n\nQuestion: {user_query} \n\nAnswer:"""
        
        messages=[
            {"role": "user", "content": generate_prompt}
        ]
        
        response = llm.invoke(messages)

        print(f"\n--- Response from R1 ---")
        print(response.content)
        print("\n" + "-"*50 + "\n")



'''
generate_prompt = f"""Please answer the following question based on the context provided. 

        Before answering, analyze each document in the context and identify if it contains the answer to the question. 
        Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
        Also, make sure to list the most relevant documents first and then answer the question based on those documents only.
        
        If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
        '\n\nContext:\n{context}\n\nQuestion: {user_query}"""


document = """
    Tesla, Inc. (TSLA) Financial Analysis and Market Overview - Q3 2023

    Executive Summary:
    Tesla, Inc. (NASDAQ: TSLA) continues to lead the electric vehicle (EV) market, showcasing strong financial performance and strategic growth initiatives in Q3 2023. This comprehensive analysis delves into Tesla's financial statements, market position, and future outlook, providing investors and stakeholders with crucial insights into the company's performance and potential.

    1. Financial Performance Overview:

    Revenue:
    Tesla reported total revenue of $23.35 billion in Q3 2023, marking a 9% increase year-over-year (YoY) from $21.45 billion in Q3 2022. The automotive segment remained the primary revenue driver, contributing $19.63 billion, up 5% YoY. Energy generation and storage revenue saw significant growth, reaching $1.56 billion, a 40% increase YoY.

    Profitability:
    Gross profit for Q3 2023 stood at $4.18 billion, with a gross margin of 17.9%. While this represents a decrease from the 25.1% gross margin in Q3 2022, it remains above industry averages. Operating income was $1.76 billion, resulting in an operating margin of 7.6%. Net income attributable to common stockholders was $1.85 billion, translating to diluted earnings per share (EPS) of $0.53.

    Cash Flow and Liquidity:
    Tesla's cash and cash equivalents at the end of Q3 2023 were $26.08 billion, a robust position that provides ample liquidity for ongoing operations and future investments. Free cash flow for the quarter was $0.85 billion, reflecting the company's ability to generate cash despite significant capital expenditures.

    2. Operational Highlights:

    Production and Deliveries:
    Tesla produced 430,488 vehicles in Q3 2023, a 17% increase YoY. The Model 3/Y accounted for 419,666 units, while the Model S/X contributed 10,822 units. Total deliveries reached 435,059 vehicles, up 27% YoY, demonstrating strong demand and improved production efficiency.

    Manufacturing Capacity:
    The company's installed annual vehicle production capacity increased to over 2 million units across its factories in Fremont, Shanghai, Berlin-Brandenburg, and Texas. The Shanghai Gigafactory remains the highest-volume plant, with an annual capacity exceeding 950,000 units.

    Energy Business:
    Tesla's energy storage deployments grew by 90% YoY, reaching 4.0 GWh in Q3 2023. Solar deployments also increased by 48% YoY to 106 MW, reflecting growing demand for Tesla's energy products.

    3. Market Position and Competitive Landscape:

    Global EV Market Share:
    Tesla maintained its position as the world's largest EV manufacturer by volume, with an estimated global market share of 18% in Q3 2023. However, competition is intensifying, particularly from Chinese manufacturers like BYD and established automakers accelerating their EV strategies.

    Brand Strength:
    Tesla's brand value continues to grow, ranked as the 12th most valuable brand globally by Interbrand in 2023, with an estimated brand value of $56.3 billion, up 4% from 2022.

    Technology Leadership:
    The company's focus on innovation, particularly in battery technology and autonomous driving capabilities, remains a key differentiator. Tesla's Full Self-Driving (FSD) beta program has expanded to over 800,000 customers in North America, showcasing its advanced driver assistance systems.

    4. Strategic Initiatives and Future Outlook:

    Product Roadmap:
    Tesla reaffirmed its commitment to launching the Cybertruck in 2023, with initial deliveries expected in Q4. The company also hinted at progress on a next-generation vehicle platform, aimed at significantly reducing production costs.

    Expansion Plans:
    Plans for a new Gigafactory in Mexico are progressing, with production expected to commence in 2025. This facility will focus on producing Tesla's next-generation vehicles and expand the company's North American manufacturing footprint.

    Battery Production:
    Tesla continues to ramp up its in-house battery cell production, with 4680 cells now being used in Model Y vehicles produced at the Texas Gigafactory. The company aims to achieve an annual production rate of 1,000 GWh by 2030.

    5. Risk Factors and Challenges:

    Supply Chain Constraints:
    While easing compared to previous years, supply chain issues continue to pose challenges, particularly in sourcing semiconductor chips and raw materials for batteries.

    Regulatory Environment:
    Evolving regulations around EVs, autonomous driving, and data privacy across different markets could impact Tesla's operations and expansion plans.

    Macroeconomic Factors:
    Rising interest rates and inflationary pressures may affect consumer demand for EVs and impact Tesla's profit margins.

    Competition:
    Intensifying competition in the EV market, especially in key markets like China and Europe, could pressure Tesla's market share and pricing power.

    6. Financial Ratios and Metrics:

    Profitability Ratios:
    - Return on Equity (ROE): 18.2%
    - Return on Assets (ROA): 10.3%
    - EBITDA Margin: 15.7%

    Liquidity Ratios:
    - Current Ratio: 1.73
    - Quick Ratio: 1.25

    Efficiency Ratios:
    - Asset Turnover Ratio: 0.88
    - Inventory Turnover Ratio: 11.2

    Valuation Metrics:
    - Price-to-Earnings (P/E) Ratio: 70.5
    - Price-to-Sales (P/S) Ratio: 7.8
    - Enterprise Value to EBITDA (EV/EBITDA): 41.2

    7. Segment Analysis:

    Automotive Segment:
    - Revenue: $19.63 billion (84% of total revenue)
    - Gross Margin: 18.9%
    - Key Products: Model 3, Model Y, Model S, Model X

    Energy Generation and Storage:
    - Revenue: $1.56 billion (7% of total revenue)
    - Gross Margin: 14.2%
    - Key Products: Powerwall, Powerpack, Megapack, Solar Roof

    Services and Other:
    - Revenue: $2.16 billion (9% of total revenue)
    - Gross Margin: 5.3%
    - Includes vehicle maintenance, repair, and used vehicle sales

    8. Geographic Revenue Distribution:

    - United States: $12.34 billion (53% of total revenue)
    - China: $4.67 billion (20% of total revenue)
    - Europe: $3.97 billion (17% of total revenue)
    - Other: $2.37 billion (10% of total revenue)

    9. Research and Development:

    Tesla invested $1.16 billion in R&D during Q3 2023, representing 5% of total revenue. Key focus areas include:
    - Next-generation vehicle platform development
    - Advancements in battery technology and production processes
    - Enhancements to Full Self-Driving (FSD) capabilities
    - Energy storage and solar technology improvements

    10. Capital Expenditures and Investments:

    Capital expenditures for Q3 2023 totaled $2.46 billion, primarily allocated to:
    - Expansion and upgrades of production facilities
    - Tooling for new products, including the Cybertruck
    - Supercharger network expansion
    - Investments in battery cell production capacity

    11. Debt and Capital Structure:

    As of September 30, 2023:
    - Total Debt: $5.62 billion
    - Total Equity: $43.51 billion
    - Debt-to-Equity Ratio: 0.13
    - Weighted Average Cost of Capital (WACC): 8.7%

    12. Stock Performance and Shareholder Returns:

    - 52-Week Price Range: $152.37 - $299.29
    - Market Capitalization: $792.5 billion (as of October 31, 2023)
    - Dividend Policy: Tesla does not currently pay dividends, reinvesting profits into growth initiatives
    - Share Repurchases: No significant share repurchases in Q3 2023

    13. Corporate Governance and Sustainability:

    Board Composition:
    Tesla's Board of Directors consists of 8 members, with 6 independent directors. The roles of CEO and Chairman are separate, with Robyn Denholm serving as Chairwoman.

    ESG Initiatives:
    - Environmental: Committed to using 100% renewable energy in all operations by 2030
    - Social: Focus on diversity and inclusion, with women representing 29% of the global workforce
    - Governance: Enhanced transparency in supply chain management and ethical sourcing of materials

    14. Analyst Recommendations and Price Targets:

    As of October 31, 2023:
    - Buy: 22 analysts
    - Hold: 15 analysts
    - Sell: 5 analysts
    - Average 12-month price target: $245.67

    15. Upcoming Catalysts and Events:

    - Cybertruck production ramp-up and initial deliveries (Q4 2023)
    - Investor Day 2024 (Date TBA)
    - Potential unveiling of next-generation vehicle platform (2024)
    - Expansion of FSD beta program to additional markets

    Conclusion:
    Tesla's Q3 2023 financial results demonstrate the company's continued leadership in the EV market, with strong revenue growth and operational improvements. While facing increased competition and margin pressures, Tesla's robust balance sheet, technological innovations, and expanding product portfolio position it well for future growth. Investors should monitor key metrics such as production ramp-up, margin trends, and progress on strategic initiatives to assess Tesla's long-term value proposition in the rapidly evolving automotive and energy markets.
"""

def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        Based on the following information, please provide a concise and accurate answer to the question.
        If the information is not sufficient to answer the question, say so.

        Question: {query}

        Relevant information:
        {chunks}

        Answer:
        """)
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = self.llm.invoke(messages)
        return response.content



for query in queries:
        print(f"\nQuery: {query}")

        # Retrieve from original vectorstore
        original_vector_results = original_vectorstore.similarity_search(query, k=3)

        # Retrieve from contextualized vectorstore
        contextualized_vector_results = contextualized_vectorstore.similarity_search(query, k=3)

        # Retrieve from original BM25
        original_tokenized_query = query.split()
        original_bm25_results = original_bm25_index.get_top_n(original_tokenized_query, original_chunks, n=3)

        # Retrieve from contextualized BM25
        contextualized_tokenized_query = query.split()
        contextualized_bm25_results = contextualized_bm25_index.get_top_n(contextualized_tokenized_query, contextualized_chunks, n=3)

        # Generate answers
        original_vector_answer = cr.generate_answer(query, [doc.page_content for doc in original_vector_results])
        contextualized_vector_answer = cr.generate_answer(query, [doc.page_content for doc in contextualized_vector_results])
        original_bm25_answer = cr.generate_answer(query, [doc.page_content for doc in original_bm25_results])
        contextualized_bm25_answer = cr.generate_answer(query, [doc.page_content for doc in contextualized_bm25_results])


        print("\nOriginal Vector Search Results:")
        for i, doc in enumerate(original_vector_results, 1):
            print(f"{i}. {doc.page_content[:200]}...")

        print("\nOriginal Vector Search Answer:")
        print(original_vector_answer)
        print("\n" + "-"*50)

        print("\nContextualized Vector Search Results:")
        for i, doc in enumerate(contextualized_vector_results, 1):
            print(f"{i}. {doc.page_content[:200]}...")

        print("\nContextualized Vector Search Answer:")
        print(contextualized_vector_answer)
        print("\n" + "-"*50)

        print("\nOriginal BM25 Search Results:")
        for i, doc in enumerate(original_bm25_results, 1):
            print(f"{i}. {doc.page_content[:200]}...")

        print("\nOriginal BM25 Search Answer:")
        print(original_bm25_answer)
        print("\n" + "-"*50)

        print("\nContextualized BM25 Search Results:")
        for i, doc in enumerate(contextualized_bm25_results, 1):
            print(f"{i}. {doc.page_content[:200]}...")

        print("\nContextualized BM25 Search Answer:")
        print(contextualized_bm25_answer)

        print("\n" + "="*50)
'''