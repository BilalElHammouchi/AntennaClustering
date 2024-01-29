# AntennaClustering

AntennaClustering is a web application for performing unsupervised learning on datasets using the ant clustering algorithm. The application allows users to upload a CSV file, explore the dataset, configure clustering parameters, and visualize the clustering results in real-time.
![ants](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/afc3889d-87df-4743-abb8-ebe0134cb7cb)

## Features

- **Upload CSV**: Users can upload a CSV file containing their dataset.

<div align="center">
  
  ![Screenshot 2024-01-29 004348](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/c0783501-ef02-45cb-ac51-bb776873d27d)
</div>

- **Data Exploration**: Provides basic insights into the dataset such as the number of columns, rows, and a summary of numerical columns.

<div align="center">
  
  ![Screenshot 2024-01-29 004455](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/741a3241-ec83-4f23-9c4e-d19e3f6936ed)
</div>

- **Visualizations**: Offers visualizations for categorical columns using pie charts and allows users to reduce the dimensionality of the dataset for visualization using t-SNE.

<div align="center">
  
![Screenshot 2024-01-29 004604](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/640d4edd-8507-4388-9da9-71e2c413f02c)
</div>

- **Clustering Configuration**: Users can set the number of clusters, ants, iterations, and threads for the ant clustering algorithm.

<div align="center">  
  
![Screenshot 2024-01-29 004650](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/1c4033e3-781a-4390-b3c8-c7f414b1b353)
</div>

- **Real-time Visualization**: Displays a scatter plot showcasing the clustering process and updates it after each iteration to visualize the algorithm's learning over time.

<div align="center">

![0129](https://github.com/GoldenDovah/AntennaClustering/assets/19519174/2b1acdf8-001f-4726-8428-3a48e1550f37)
</div>

- **Interactive Interface**: Offers an intuitive user interface built with Streamlit for easy navigation and interaction.

## Usage

To run the application locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your_username/AntennaClustering.git
```

2. Navigate to the project directory:

```bash
cd AntennaClustering
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run webapp.py
```

## Dependencies
The project relies on the following Python libraries:

- streamlit
- pandas
- matplotlib
- plotly
- seaborn
- scikit-learn
- numpy
- tqdm
