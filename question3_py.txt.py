# Libraries for decision tree classification training
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Libraries for decision tree visualization
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

# Plotting graph of metrics
import matplotlib.pyplot as plt


def generate_decision_tree(df,highest_corr_var,min_leaf_node,decision_tree_image):
    
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=80/310, random_state=1)
    
    # Create a decision tree classifier with entropy as the criterion
    clf = DecisionTreeClassifier(min_samples_leaf=min_leaf_node,criterion='gini',splitter="best")

    # Train the model
    clf=clf.fit(X_train,y_train)

    # Predict the testing data results using the model
    y_pred = clf.predict(X_test)
    
    features = ["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"]
    features.remove(highest_corr_var)
    target_names = ['Normal','Abnormal']
    
    # Metrics for analysis of the prediction
    metric_dict = metrics.classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
    print(metric_dict)
    
    accuracy_list.append(metric_dict['accuracy'])
    precision_normal.append(metric_dict['Normal']['precision'])
    precision_abnormal.append(metric_dict['Abnormal']['precision'])
    recall_normal.append(metric_dict['Normal']['recall'])
    recall_abnormal.append(metric_dict['Abnormal']['recall'])
    
    
    # Visualize the decision tree
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = features,class_names=['Normal','Abnormal'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(decision_tree_image+".png")
    Image(graph.create_png())
    

def plotGraph():
    x_axis = ['5','15','25','40','50']
    metric_df = pd.DataFrame(index = x_axis)
    
    metric_df['Accuracy'] = accuracy_list
    metric_df['Precision_Normal']=precision_normal
    metric_df['Precision_Abnormal']=precision_abnormal
    metric_df['Recall_Normal']=recall_normal
    metric_df['Recall_Abnormal']=recall_abnormal
    
    metric_plot = metric_df.plot()
    metric_plot.set_xlabel("Minimum Number of leaf nodes")
    metric_plot.set_ylabel("Metric values")
    metric_plot.legend(bbox_to_anchor=(1.2, 0.5))    
    
    metric_plot.get_figure().savefig("question3_graph.png")

def do_corr_and_prepare_dataframe():
    
    # Read the data
    biomed_data2 = pd.read_csv("Biomechanical_Data_2Classes.csv",header=0)

    # Randomly shuffle the data
    biomed_data2 = biomed_data2.sample(frac=1).reset_index(drop=True)

    # Binarize the class values
    biomed_data2["class"] = biomed_data2["class"].astype('category').cat.codes

    # Find the correlation between 'Class' and the remaining variables
    corr=biomed_data2[biomed_data2.columns[0:]].corr()['class'][:]

    # Drop the 'class' column
    corr = corr.drop(labels=['class'])

    # Take the absolute value of the series
    corr = corr.abs()

    # Find the index(attribute/variable) containing the max element
    highest_corr_var = corr.idxmax()

    # Drop the highest correlation column from the dataframe
    biomed_data2_afterdrop = biomed_data2.drop(columns=[highest_corr_var])
    
    return biomed_data2_afterdrop,highest_corr_var


# Initialize lists for metrics
accuracy_list    = list()
precision_normal = list()
precision_abnormal = list()
recall_normal    = list()
recall_abnormal    = list()

# Get dataframe after correlation function
biomed_data2_afterdrop, highest_corr_var = do_corr_and_prepare_dataframe()

# Generate Decision trees with different minimum leaf nodes
generate_decision_tree(biomed_data2_afterdrop,highest_corr_var,5,"question3_5")
generate_decision_tree(biomed_data2_afterdrop,highest_corr_var,15,"question3_15")
generate_decision_tree(biomed_data2_afterdrop,highest_corr_var,25,"question3_25")
generate_decision_tree(biomed_data2_afterdrop,highest_corr_var,40,"question3_40")
generate_decision_tree(biomed_data2_afterdrop,highest_corr_var,50,"question3_50")

# Plot graphs of min. leaf nodes against metrics    
plotGraph()    
