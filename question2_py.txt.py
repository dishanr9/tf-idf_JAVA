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
import matplotlib.pyplot as pltgenerate_decision_tree


def generate_decision_tree(min_sample_leaf,decision_tree_image):
    
    X_train, X_test, y_train, y_test = train_test_split(biomed_data3.iloc[:,:-1], biomed_data3.iloc[:,-1], test_size=80/310, random_state=1)
    
    # Create a decision tree classifier with entropy as the criterion
    clf = DecisionTreeClassifier(min_samples_leaf=min_sample_leaf,criterion='gini',splitter="best")

    # Train the model
    clf=clf.fit(X_train,y_train)

    # Predict the testing data results using the model
    y_pred = clf.predict(X_test)
    
    features = ["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"]
    target_names = ['Normal','Hernia','Spondylolisthesis']
    
    # Metrics for analysis of the prediction
    metric_dict = metrics.classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
    
    accuracy_list.append(metric_dict['accuracy'])
    precision_normal.append(metric_dict['Normal']['precision'])
    precision_hernia.append(metric_dict['Hernia']['precision'])
    precision_spondy.append(metric_dict['Spondylolisthesis']['precision'])
    recall_normal.append(metric_dict['Normal']['recall'])
    recall_hernia.append(metric_dict['Hernia']['recall'])
    recall_spondy.append(metric_dict['Spondylolisthesis']['recall'])
    
    
    # Visualize the decision tree
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = features,class_names=['Normal','Hernia','Spondylolisthesis'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(decision_tree_image+".png")
    Image(graph.create_png())
    

def plotGraph():
    x_axis = ['5','15','25','40','50']
    metric_df = pd.DataFrame(index = x_axis)
    
    metric_df['Accuracy'] = accuracy_list
    metric_df['Precision_Normal']=precision_normal
    metric_df['Precision_Hernia']=precision_hernia
    metric_df['Precision_Spondylolisthesis']=precision_spondy
    metric_df['Recall_Normal']=recall_normal
    metric_df['Recall_Hernia']=recall_hernia
    metric_df['Recall_Spondylolisthesis']=recall_spondy
    
    metric_plot = metric_df.plot()
    metric_plot.set_xlabel("Minimum Number of leaf nodes")
    metric_plot.set_ylabel("Metric values")
    metric_plot.legend(bbox_to_anchor=(1.2, 0.5))    
    
    metric_plot.get_figure().savefig("question2_graph.png")


# Read the data
biomed_data3 = pd.read_csv("Biomechanical_Data_3Classes.csv",header=0)

# Randomly shuffle the data
biomed_data3 = biomed_data3.sample(frac=1).reset_index(drop=True)
   
accuracy_list    = list()
precision_hernia = list()
precision_normal = list()
precision_spondy = list()
recall_hernia    = list()
recall_normal    = list()
recall_spondy    = list()

# Generate Decision trees with different minimum leaf nodes
generate_decision_tree(5,"question2_5")
generate_decision_tree(15,"question2_15")
generate_decision_tree(25,"question2_25")
generate_decision_tree(40,"question2_40")
generate_decision_tree(50,"question2_50")

# Plot graphs of min. leaf nodes against metrics    
plotGraph()
    


