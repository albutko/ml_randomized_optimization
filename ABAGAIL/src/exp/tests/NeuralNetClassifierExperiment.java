package exp.tests;

import opt.*;
import opt.ga.*;
import shared.*;
import shared.filt.*;
import shared.reader.*;
import shared.tester.*;
import func.nn.backprop.*;

import java.text.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;



public class NeuralNetClassifierExperiment {

    private static int inputLayer = 18, hiddenLayer1 = 100, hiddenLayer2 = 50, outputLayer = 1, trainingIterations = 500;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();


    private static DataSet set = getDataSet();

    private static double[] labels = {0,1};
    private static MyRecallTestMetric recall = new MyRecallTestMetric();


    private static TestMetric[][] metrics =  {{new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()},
            {new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()},
            {new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()}};

    private static TestTrainSplitFilter ttsf = new TestTrainSplitFilter(75);


    private static DataSet trainSet = null;
    private static DataSet testSet = null;

    private static Instance[] trainInstances = null;
    private static Instance[] testInstances = null;


    private static MyNeuralNetworkTester[] nnTester = new MyNeuralNetworkTester[2];
    private static BackPropagationNetwork[] networks = new BackPropagationNetwork[2];
    private static NeuralNetworkOptimizationProblemRecall[] nnop = new NeuralNetworkOptimizationProblemRecall[2];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[2];

    private static String[] oaNames = {"RHC", "SA", "GA"};
//    private static String[] oaNames = {"RHC"};


    private static String[] results = new String[2];
    private static String csvResults = new String();

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        ttsf.filter(set);
        trainSet = ttsf.getTrainingSet();
        testSet = ttsf.getTestingSet();

        trainInstances = trainSet.getInstances();
        testInstances = testSet.getInstances();

        for(int iter = 0; iter < 5; iter++) {
            TestMetric[][] metrics =  {{new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()}, {new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()}, {new MyConfusionMatrixTestMetric(labels), new MyRecallTestMetric()}};

            nnTester = new MyNeuralNetworkTester[2];
            networks = new BackPropagationNetwork[2];
            nnop = new NeuralNetworkOptimizationProblemRecall[2];
            oa = new OptimizationAlgorithm[2];
            csvResults = new String();
            String[] results = new String[2];

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblemRecall(trainSet, networks[i], recall);
        }

        oa[0] = new MyRandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E3, .95, nnop[1]);
//        oa[2] = new StandardGeneticAlgorithm(100, 50, 10, nnop[2]);

            for (int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i]); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[i].getOptimal();

                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();

                for (int j = 0; j < testInstances.length; j++) {
                    networks[i].setInputValues(testInstances[j].getData());
                    networks[i].run();


                    predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }

                end = System.nanoTime();
                nnTester[i] = new MyNeuralNetworkTester(networks[i], metrics[i]);
                nnTester[i].test(testInstances);


                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results[i] = "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            }


            for (int i = 0; i < oa.length; i++) {
                TestMetric[] mets = nnTester[i].getMetrics();
                System.out.println(results[i]);
                for (int j = 0; j < mets.length; j++) {
                    mets[j].printResults();
                }
            }

            try {
                writeStringToCSV("../nn_training_log" + iter + ".csv", csvResults);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nRecall results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            double val = oa.train();

            csvResults += oaName+","+i+"," + val +"\n";
            System.out.println("Iteration " + i+ ": " + val);
        }
    }

    private static DataSet getDataSet() {

        CSVDataSetReader reader = new CSVDataSetReader("src/exp/tests/higgs_train.txt");
        DataSet ds = null;
        try {
            ds = reader.read();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        return ds;
    }

    private static void writeStringToCSV(String fileName, String s) throws IOException {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, false));
            writer.append(s);

            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

}

