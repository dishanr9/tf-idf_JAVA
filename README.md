# tf-idf using Hadoop (JAVA) 
CS7070-Big Data Analytics
Spring 2020
Programming Assignment #1
Due Date:  March 9th, 2020 (9PM)

In this programming assignment, you are expected to use and modify the MapReduce programs for computing the TFIDF for terms in a set of documents. Use "input_tfidf.txt" as your input file. The tasks to be accomplished by, as parts of this assignment,  you are:

1.	(20) Execute all phases of the TFIDF program, on the small sample data shared by Sahil, and submit the following items:
a.	TFIDF for top 18 terms in each document, sorted in descending order of their tfidf values, and formatted for easy readability.

2.	 (25) Modify the programs to remove from consideration all those words that occur only once or twice in each document. Repeat the task of Q1 above. 

a. Comment on any changes in the results of part 1(a). 
b. Select at least 3 different words for which there is a change in their tfidf values and explain the reason for the change.

3.	(30) Now consider a “Term” to mean a 2-gram (two words occurring sequentially) in a document. Modify the programs given to you to compute the TFIDF for each 2-gram. Submit the following items:
a.	List of top 20 2-grams for each document, having the highest TFIDF values. The task of selecting the top 20 terms does not need to be done by the MapReduce program.
b.	Which output – obtained in 3(a) or in 2(a) – better characterizes the documents? Give reasons for your answers.

4.	(20) Once your program is working for the above two parts, run the programs on a larger collection of documents (to be provided to you by March 2nd) and repeat the above task . Discuss the results for 1(a), 2(a), and 3(a) in the context of the new set of documents. 

5.	(5) Well organized and clearly understandable presentation of results in the submission.


Your submission for each part must include:
•	Listing of source code used
•	Output produced by your code
•	A description of input files used and any parameters etc. set in your program.
•	Interpretation of or comments about the results obtained by your program.


