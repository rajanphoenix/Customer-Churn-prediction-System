import plotly.express as px

# Our standard SaaS color palette for risk
RISK_COLORS = {'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e74c3c'}

def plot_categorical_distribution(df, column_name):
    """Left Chart: Shows the total count of customers in a category."""
    counts = df[column_name].value_counts().reset_index()
    counts.columns = [column_name, 'Customer Count']
    
    fig = px.bar(counts, x=column_name, y='Customer Count', 
                 text_auto=True, title=f"Customer Distribution by {column_name}")
    fig.update_layout(xaxis_title="", showlegend=False)
    return fig

def plot_categorical_risk(df, column_name):
    """Right Chart: Shows the % of High/Med/Low risk within each category."""
    # 1. Get raw counts
    risk_dist = df.groupby([column_name, 'Risk_Level']).size().reset_index(name='Count')
    
    # 2. Calculate explicit percentages safely in Pandas
    total_per_cat = risk_dist.groupby(column_name)['Count'].transform('sum')
    risk_dist['Percentage'] = (risk_dist['Count'] / total_per_cat) * 100
    
    # 3. Plot the explicit percentages
    fig = px.bar(risk_dist, x=column_name, y='Percentage', color='Risk_Level',
                 color_discrete_map=RISK_COLORS, barmode='stack',
                 category_orders={"Risk_Level": ["High Risk", "Medium Risk", "Low Risk"]},
                 title=f"Risk Proportion by {column_name} (%)")
                 
    fig.update_layout(xaxis_title="", yaxis_title="Percentage (%)")
    return fig

def plot_numerical_distribution(df, column_name):
    """Left Chart: Histogram for numbers like Tenure or Seats."""
    fig = px.histogram(df, x=column_name, nbins=20, 
                       title=f"Distribution of {column_name}")
    fig.update_layout(yaxis_title="Number of Accounts")
    return fig

def plot_numerical_risk(df, column_name):
    """Right Chart: Scatter plot of the number vs actual Churn Probability."""
    fig = px.scatter(df, x=column_name, y='Churn_Probability', 
                     color='Risk_Level', color_discrete_map=RISK_COLORS,
                     title=f"Risk Correlation: {column_name}")
    # Draw the High Risk threshold line
    fig.add_hline(y=0.354, line_dash="dash", line_color="red", annotation_text="High Risk")
    return fig