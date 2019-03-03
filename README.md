## Running My Code
This code is based fully on the [ABAGAIL](https://github.com/pushkar/ABAGAIL) repository and has similar 
build and run instructions as ABAGAIL.

### Requirements
* Java 8
* ant

### Instructions
1. Create ABAGAIL.jar
    * Navigate to the ABAGAIL directory and run ant to build the jar file
        * `cd ABAGAIL`
        * `ant`
2. Run Tests within the `ABAGAIL.jar` using java:
    * In the root directory run the below command to run a specific class
        * `java -cp ABAGAIL.jar path.to.class`
        
### Tests
There are 4 tests to run in this assignment. Below are the commands of how to run them and their names
* NeuralNetClassifierExperiment
    * `java -cp ABAGAIL.jar exp.tests.NeuralNetClassifierExperiment`
* MyFourPeaksTest
    * `java -cp ABAGAIL.jar exp.tests.MyFourPeaksTest`
* MyKnapsackTest
    * `java -cp ABAGAIL.jar exp.tests.MyKnapsackTest`
* MyTravelingSalesmanTest
    * `java -cp ABAGAIL.jar exp.tests.MyTravelingSalesmanTest`
    
In each of the optimization problems you can change the number of iterations alotted to each algorithm to see
performance based on fixed iterations. 