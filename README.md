# RFM-analysis
RFM analysis customer segmentation
1. The code begins with the import of necessary libraries such as Pandas for data manipulation and analysis, as well as datetime for date and time operations.

2. A CSV file "Sample - Superstore.csv" is read into a pandas DataFrame using the `read_csv` method to analyze the data.

3. The dataset is preprocessed and transformed to create the RFM (Recency, Frequency, Monetary) values for each customer.

4. Customers are segmented using the RFM values and quartiles to categorize them into different segments based on their buying behavior and spending patterns.

5. The data is further analyzed and visualized using various techniques such as distribution plots, log transformation to normalize the skewed data, and the creation of value segments based on RFM scores.

6. The code uses Pandas `qcut` to allocate customers to value segments based on their RFM scores, and then it visualizes the distribution of these segments using a bar plot.

7. Finally, the code creates customer segments based on their RFM scores and counts the customers in each segment. It then generates a treemap visualization using Plotly Express to represent the hierarchical nature of the segments and their distribution.  
