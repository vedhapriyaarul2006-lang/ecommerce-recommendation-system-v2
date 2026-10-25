import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Personalized Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# ==================================================
# PAGE STYLE
# ==================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.header-box {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-bottom: 25px;
}

.header-box h1 {
    font-size: 32px;
}

.recommendation-box {
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #444444;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="header-box">
    <h1>🛍️ Personalized Product Recommendation System</h1>
    <p>
    Collaborative Filtering + Content-Based Filtering +
    Hybrid Recommendation + Dynamic User Interest +
    Cold-Start Recommendation + Explainable AI
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATASET
# ==================================================

FILE_PATH = "ecommerce_sales_analytics_with_products.csv"

try:
    df = pd.read_csv(FILE_PATH)

except FileNotFoundError:
    st.error(
        f"Dataset not found: {FILE_PATH}. "
        "Keep the CSV file in the same folder as app.py."
    )
    st.stop()

# Remove spaces from column names
df.columns = df.columns.str.strip()

# ==================================================
# CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "order_id",
    "order_date",
    "customer_id",
    "product_category",
    "quantity",
    "unit_price",
    "customer_rating",
    "revenue",
    "ProductID",
    "ProductName"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"The following columns are missing: {missing_columns}"
    )

    st.write("Available columns:")
    st.write(list(df.columns))

    st.info(
        "Check whether ProductID and ProductName are spelled "
        "exactly the same in your CSV file."
    )

    st.stop()

# ==================================================
# DATA PREPROCESSING
# ==================================================

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

numeric_columns = [
    "quantity",
    "unit_price",
    "customer_rating",
    "revenue"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "customer_id",
        "ProductID",
        "ProductName",
        "product_category"
    ]
)

df["customer_id"] = df["customer_id"].astype(str)
df["ProductID"] = df["ProductID"].astype(str)
df["ProductName"] = df["ProductName"].astype(str)
df["product_category"] = df["product_category"].astype(str)

# ==================================================
# SIDEBAR SETTINGS
# ==================================================

st.sidebar.title("⚙️ Recommendation Settings")

customer_list = sorted(
    df["customer_id"].unique().tolist()
)

selected_customer = st.sidebar.selectbox(
    "Select Customer",
    customer_list
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

content_weight = st.sidebar.slider(
    "Content-Based Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

collaborative_weight = 1 - content_weight

st.sidebar.write(
    f"Collaborative Filtering Weight: "
    f"{collaborative_weight:.1f}"
)

st.sidebar.write(
    f"Content-Based Filtering Weight: "
    f"{content_weight:.1f}"
)

# ==================================================
# PRODUCT INFORMATION TABLE
# ==================================================

product_information = df.groupby(
    [
        "ProductID",
        "ProductName",
        "product_category"
    ]
).agg(
    Average_Rating=("customer_rating", "mean"),
    Total_Quantity=("quantity", "sum"),
    Total_Revenue=("revenue", "sum"),
    Average_Price=("unit_price", "mean")
).reset_index()

# ==================================================
# CUSTOMER-PRODUCT MATRIX
# ==================================================

customer_product_matrix = pd.pivot_table(
    df,
    index="customer_id",
    columns="ProductID",
    values="quantity",
    aggfunc="sum",
    fill_value=0
)

if selected_customer not in customer_product_matrix.index:
    st.warning(
        "This customer does not have enough purchase history."
    )
    st.stop()

# ==================================================
# COLLABORATIVE FILTERING
# ==================================================

def collaborative_filtering(customer_id):

    similarity_matrix = cosine_similarity(
        customer_product_matrix
    )

    customer_index = customer_product_matrix.index.get_loc(
        customer_id
    )

    customer_similarities = similarity_matrix[
        customer_index
    ]

    similarity_series = pd.Series(
        customer_similarities,
        index=customer_product_matrix.index
    )

    # Remove the selected customer
    similarity_series = similarity_series.drop(
        customer_id,
        errors="ignore"
    )

    # Select the 10 most similar customers
    similar_customers = similarity_series.sort_values(
        ascending=False
    ).head(10)

    recommendation_scores = pd.Series(
        0.0,
        index=customer_product_matrix.columns
    )

    for other_customer, similarity_score in similar_customers.items():

        other_customer_products = customer_product_matrix.loc[
            other_customer
        ]

        recommendation_scores += (
            other_customer_products * similarity_score
        )

    if recommendation_scores.max() > 0:
        recommendation_scores = (
            recommendation_scores /
            recommendation_scores.max()
        )

    return recommendation_scores

# ==================================================
# CONTENT-BASED FILTERING
# ==================================================

def content_based_filtering(customer_id):

    customer_history = df[
        df["customer_id"] == customer_id
    ]

    # Customer's preferred categories
    category_preference = customer_history.groupby(
        "product_category"
    )["quantity"].sum()

    # Product-level information
    product_scores = product_information.copy()

    # Normalize product quantity, rating and revenue
    scaler = MinMaxScaler()

    product_scores["quantity_score"] = scaler.fit_transform(
        product_scores[["Total_Quantity"]]
    )

    product_scores["rating_score"] = scaler.fit_transform(
        product_scores[["Average_Rating"]]
    )

    product_scores["revenue_score"] = scaler.fit_transform(
        product_scores[["Total_Revenue"]]
    )

    # Category preference score
    product_scores["category_preference"] = (
        product_scores["product_category"].map(
            category_preference
        ).fillna(0)
    )

    if product_scores["category_preference"].max() > 0:
        product_scores["category_preference"] = (
            product_scores["category_preference"]
            /
            product_scores["category_preference"].max()
        )

    # Content-based score
    product_scores["content_score"] = (
        product_scores["category_preference"] * 0.5
        +
        product_scores["rating_score"] * 0.2
        +
        product_scores["quantity_score"] * 0.2
        +
        product_scores["revenue_score"] * 0.1
    )

    content_scores = product_scores.set_index(
        "ProductID"
    )["content_score"]

    content_scores = content_scores.reindex(
        customer_product_matrix.columns,
        fill_value=0
    )

    if content_scores.max() > 0:
        content_scores = (
            content_scores /
            content_scores.max()
        )

    return content_scores

# ==================================================
# HYBRID RECOMMENDATION
# ==================================================

def hybrid_recommendation(customer_id):

    collaborative_scores = collaborative_filtering(
        customer_id
    )

    content_scores = content_based_filtering(
        customer_id
    )

    hybrid_scores = (
        collaborative_scores * collaborative_weight
        +
        content_scores * content_weight
    )

    # Products already purchased by the customer
    purchased_products = set(
        df[
            df["customer_id"] == customer_id
        ]["ProductID"]
    )

    # Remove already purchased products
    hybrid_scores = hybrid_scores.drop(
        labels=list(purchased_products),
        errors="ignore"
    )

    # If all products were purchased, show top products
    if hybrid_scores.empty:

        hybrid_scores = (
            collaborative_scores * collaborative_weight
            +
            content_scores * content_weight
        )

    return hybrid_scores.sort_values(
        ascending=False
    ).head(number_of_recommendations)

# ==================================================
# RECOMMENDATION REASON
# ==================================================

def get_recommendation_reason(
    customer_id,
    product_id
):

    customer_history = df[
        df["customer_id"] == customer_id
    ]

    recommended_product = product_information[
        product_information["ProductID"] == product_id
    ]

    if recommended_product.empty:
        return "Recommended based on recommendation score."

    recommended_category = (
        recommended_product["product_category"].iloc[0]
    )

    recommended_rating = (
        recommended_product["Average_Rating"].iloc[0]
    )

    recommended_quantity = (
        recommended_product["Total_Quantity"].iloc[0]
    )

    reasons = []

    # Reason 1: Customer category interest
    purchased_categories = (
        customer_history["product_category"]
        .value_counts()
    )

    if recommended_category in purchased_categories.index:

        reasons.append(
            f"you previously purchased products from "
            f"the {recommended_category} category"
        )

    # Reason 2: Similar customers
    collaborative_scores = collaborative_filtering(
        customer_id
    )

    if product_id in collaborative_scores.index:
        if collaborative_scores[product_id] > 0:

            reasons.append(
                "customers with similar purchase patterns "
                "also purchased this product"
            )

    # Reason 3: Product rating
    if recommended_rating >= 4:

        reasons.append(
            f"this product has a good average rating "
            f"of {recommended_rating:.1f}"
        )

    # Reason 4: Popularity
    if recommended_quantity > df["quantity"].median():

        reasons.append(
            "this product is popular in the dataset"
        )

    # Reason 5: Default reason
    if not reasons:

        reasons.append(
            "the hybrid recommendation model gave "
            "this product a high score"
        )

    reason_text = "; ".join(reasons)

    return "Recommended because " + reason_text + "."

# ==================================================
# COLD-START RECOMMENDATION
# ==================================================

def cold_start_recommendation():

    cold_start_data = product_information.copy()

    scaler = MinMaxScaler()

    cold_start_data["quantity_score"] = scaler.fit_transform(
        cold_start_data[["Total_Quantity"]]
    )

    cold_start_data["rating_score"] = scaler.fit_transform(
        cold_start_data[["Average_Rating"]]
    )

    cold_start_data["revenue_score"] = scaler.fit_transform(
        cold_start_data[["Total_Revenue"]]
    )

    cold_start_data["cold_start_score"] = (
        cold_start_data["quantity_score"] * 0.4
        +
        cold_start_data["rating_score"] * 0.4
        +
        cold_start_data["revenue_score"] * 0.2
    )

    return cold_start_data.sort_values(
        "cold_start_score",
        ascending=False
    )

# ==================================================
# EVALUATION
# ==================================================

def evaluate_recommendations():

    evaluation_results = []

    customers = df[
        "customer_id"
    ].unique()

    for customer_id in customers[:100]:

        purchased_products = set(
            df[
                df["customer_id"] == customer_id
            ]["ProductID"]
        )

        if len(purchased_products) < 2:
            continue

        recommendations = hybrid_recommendation_for_evaluation(
            customer_id
        )

        recommended_products = set(
            recommendations.index
        )

        hits = len(
            purchased_products.intersection(
                recommended_products
            )
        )

        precision = hits / max(
            len(recommended_products),
            1
        )

        recall = hits / max(
            len(purchased_products),
            1
        )

        evaluation_results.append({
            "customer_id": customer_id,
            "precision": precision,
            "recall": recall
        })

    if not evaluation_results:
        return None

    return pd.DataFrame(evaluation_results)


def hybrid_recommendation_for_evaluation(customer_id):

    collaborative_scores = collaborative_filtering(
        customer_id
    )

    content_scores = content_based_filtering(
        customer_id
    )

    hybrid_scores = (
        collaborative_scores * collaborative_weight
        +
        content_scores * content_weight
    )

    return hybrid_scores.sort_values(
        ascending=False
    ).head(5)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Customer Profile",
    "🎯 Recommendations",
    "📊 User Interest",
    "🆕 Cold Start",
    "📈 Evaluation",
    "📁 Dataset"
])

# ==================================================
# CUSTOMER PROFILE
# ==================================================

with tab1:

    st.subheader("👤 Customer Profile")

    customer_data = df[
        df["customer_id"] == selected_customer
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customer ID",
            selected_customer
        )

    with col2:
        st.metric(
            "Total Orders",
            customer_data["order_id"].nunique()
        )

    with col3:
        st.metric(
            "Products Purchased",
            customer_data["ProductID"].nunique()
        )

    with col4:
        st.metric(
            "Total Spending",
            f"₹{customer_data['revenue'].sum():,.2f}"
        )

    st.subheader("Purchase History")

    purchase_history = customer_data[
        [
            "order_id",
            "order_date",
            "ProductID",
            "ProductName",
            "product_category",
            "quantity",
            "unit_price",
            "customer_rating",
            "revenue"
        ]
    ]

    st.dataframe(
        purchase_history,
        use_container_width=True
    )

# ==================================================
# RECOMMENDATIONS
# ==================================================

with tab2:

    st.subheader("🎯 Personalized Product Recommendations")

    recommendations = hybrid_recommendation(
        selected_customer
    )

    if recommendations.empty:

        st.warning(
            "No recommendations available."
        )

    else:

        for rank, (product_id, score) in enumerate(
            recommendations.items(),
            start=1
        ):

            product_row = product_information[
                product_information["ProductID"] == product_id
            ]

            if product_row.empty:
                continue

            product_name = product_row[
                "ProductName"
            ].iloc[0]

            product_category = product_row[
                "product_category"
            ].iloc[0]

            average_rating = product_row[
                "Average_Rating"
            ].iloc[0]

            average_price = product_row[
                "Average_Price"
            ].iloc[0]

            reason = get_recommendation_reason(
                selected_customer,
                product_id
            )

            with st.container(border=True):

                col1, col2, col3 = st.columns([
                    1, 4, 2
                ])

                with col1:
                    st.markdown(f"## #{rank}")

                with col2:

                    st.markdown(
                        f"### 🛍️ {product_name}"
                    )

                    st.write(
                        f"**Product ID:** {product_id}"
                    )

                    st.write(
                        f"**Category:** {product_category}"
                    )

                    st.write(
                        f"**Average Rating:** "
                        f"{average_rating:.2f} ⭐"
                    )

                    st.write(
                        f"**Average Price:** "
                        f"₹{average_price:,.2f}"
                    )

                    st.info(
                        f"💡 **Why recommended:** {reason}"
                    )

                with col3:

                    st.metric(
                        "Recommendation Score",
                        f"{score:.3f}"
                    )

# ==================================================
# USER INTEREST
# ==================================================

with tab3:

    st.subheader("📊 Dynamic User Interest")

    interest_data = df[
        df["customer_id"] == selected_customer
    ].groupby(
        "product_category"
    )["quantity"].sum().sort_values(
        ascending=False
    )

    st.bar_chart(interest_data)

    st.write(
        "This chart shows the customer's interest "
        "based on purchased quantity."
    )

    st.subheader("Most Purchased Products")

    product_interest = df[
        df["customer_id"] == selected_customer
    ].groupby(
        ["ProductID", "ProductName"]
    )["quantity"].sum().sort_values(
        ascending=False
    )

    st.dataframe(
        product_interest.reset_index(),
        use_container_width=True
    )

# ==================================================
# COLD START
# ==================================================

with tab4:

    st.subheader("🆕 Cold-Start Recommendations")

    st.write(
        "For a new customer with no purchase history, "
        "the system recommends popular and highly rated products."
    )

    cold_start_data = cold_start_recommendation()

    st.dataframe(
        cold_start_data[
            [
                "ProductID",
                "ProductName",
                "product_category",
                "Average_Rating",
                "Total_Quantity",
                "Total_Revenue",
                "cold_start_score"
            ]
        ],
        use_container_width=True
    )

# ==================================================
# EVALUATION
# ==================================================

with tab5:

    st.subheader("📈 Recommendation Evaluation")

    if st.button("Run Evaluation"):

        with st.spinner("Evaluating recommendations..."):

            evaluation_data = evaluate_recommendations()

        if evaluation_data is None:

            st.warning(
                "Not enough customer purchase history."
            )

        else:

            average_precision = (
                evaluation_data["precision"].mean()
            )

            average_recall = (
                evaluation_data["recall"].mean()
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Average Precision",
                    f"{average_precision:.4f}"
                )

            with col2:
                st.metric(
                    "Average Recall",
                    f"{average_recall:.4f}"
                )

            st.dataframe(
                evaluation_data,
                use_container_width=True
            )

# ==================================================
# DATASET
# ==================================================

with tab6:

    st.subheader("📁 Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Customers",
            df["customer_id"].nunique()
        )

    with col4:
        st.metric(
            "Products",
            df["ProductID"].nunique()
        )

    st.subheader("Available Product Details")

    st.dataframe(
        product_information,
        use_container_width=True
    )

    with st.expander("View Complete Dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )