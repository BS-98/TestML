# TestML

## ML_CarPrices_Model.py

I am going to describe how I usually run the code:

  - 1st: 1-17 lines --> Import libraries and functions. Then I drop unnecessary (in my opinion) columns. I wondered what to do with 'city' and 'paint_color' data because maybe they could be useful for predictions but I excluded them finally. Also I delete 'size' column as it contained more than 50% of nan values.
  
  - 2nd: 20-22 --> Plot 'price' column on boxplot. I decided to delete rows where price is greater than 32555$ (the upper whisker) and less than 1$ (the lower whisker). I assumed that values out of this range, they are outliers. It is important to detect outliers because they could affect on our model.
  
  - 3rd: 24 line --> Another visualization - scatter plot of 'price' values.
  
  - 4th: 26-28 lines --> Drop rows from 2nd step.
  
  - 5th: 33 line --> It is heatmap visualization of missing values. Heatmap shows the correlation of missingness between every 2 columns. You can also uncomment 27 and 28 line to see another visualizations of missing values in the dateset. Based on heatmap we see that data: cylinders, condition, drive and type are usually present with each other.
  
  - 5th: 36-103 lines --> Filling missing values in every column.
    - 'manufacturer', 'cylinders', 'fuel', 'title_status' - filling with the most frequent value. 'Cylinders' data converted to numeric data.
    - 'make' - filling with the most frequent 'make' value for the most frequent 'manufacturer' value - Ford was the most frequent manufacturer and for the Ford the most frequent make was f-150, so missing values were filled with f-150.
    - 'year' - delete rows where year < 1900. Missing values replace with median value.
    - 'condition', 'drive', 'type' - filling nan values based on groupby function. For 'condition' based on the most frequent car's year condition. For example: Cars from 2005 were usually in 'good' condition so missing values in 'condition' where year == 2005 were filling with 'good' value. 'Drive' and 'type' missing data were filled the same way, but not based on 'year' -> based on 'manufacturer'.
    - 'odometer' missing values filled with mean values from previous year.
    - 'transmission' nan values replace with 'manual' if year < 2010, else with 'automatic'.
  
  - 6th: 107-120 lines --> Instead of making dummy variables from 'manufacturer' and 'make' data (too many unique values - 101223 for 'manufacturer' and 52 for 'make') the unique values from this columns were replaced with its count.
  
  - 7th: 123 line --> Make dummy variables from categorical data in dataset.
  
  - 8th: 125-134 lines --> Split the dataset into training set and test set. Test set = 10%.
  
  - 9th: 137-144 lines --> Scale data
  
  - 10th: 148-158 lines --> DecisionTree model with metrics calculation like: R-Squared, Mean Squared Error, Mean Absolute Error. Model saved in 'car_prices_dectree_model.sav' file.
  
  - 11th: 160-170 lines --> RandomForest model with metrics calculation like: R-Squared, Mean Squared Error, Mean Absolute Error. Model saved in 'car_prices_rndm_frst_model.sav' file.
  
  - 12th: 173-203 lines --> Artificial Neural Network - architecture: 4 deep layer with 30 neurons each layer, add dropout between each layer (0.4), ReLU activation function for deep layers and Linear activation function for output layer. Optimizer 'Adam', Loss function 'mse', metrics 'mae'. Batch size = 1024 and number of epochs = 50. As the high accuracy of model was not the main goal, I end up with combination of paramaters like this. Model saved to 'car_prices_model.h5' file. Metrics calculation like in DecisionTree model and RandomForest model was also made.
  
  - 13th: 206-212 lines --> Visualization of change mae value during the training process.
  
  - 14th: 215-221 lines --> Visualization of change loss value during the training process.
  
  
  
## Skyphrases.py 
 Just run the script.
 Answer: 410
 
## Checkerboard.py
Just run the script.
  
