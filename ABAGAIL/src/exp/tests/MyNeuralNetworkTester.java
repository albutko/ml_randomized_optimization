package exp.tests;

import func.nn.NeuralNetwork;
import shared.Instance;
import shared.reader.DataSetLabelBinarySeperator;
import shared.tester.TestMetric;
import shared.tester.Tester;

/**
 * A tester for neural networks.  This will run each instance
 * through the network and report the results to any test metrics
 * specified at instantiation.
 * 
 * @author Jesse Rosalia (https://www.github.com/theJenix)
 * @date 2013-03-05
 */
public class MyNeuralNetworkTester implements Tester {

    private NeuralNetwork network;
    private TestMetric[] metrics;

    public MyNeuralNetworkTester(NeuralNetwork network, TestMetric ... metrics) {
        this.network = network;
        this.metrics = metrics;
    }

    @Override
    public void test(Instance[] instances) {
        for (int i = 0; i < instances.length; i++) {
            //run the instance data through the network
            network.setInputValues(instances[i].getData());
            network.run();

            int actual   = new Instance(network.getOutputValues()).getDiscrete();

            //collapse the values, for statistics reporting
            //NOTE: assumes discrete labels, with n output nodes for n
            // potential labels, and an activation function that outputs
            // values between 0 and 1.
            Instance expectedOne = new Instance(instances[i].getData(), new Instance(instances[i].getLabel().getDiscrete()));
            Instance actualOne   = new Instance(instances[i].getData(), new Instance(actual));


            //run this result past all of the available test metrics
            for (TestMetric metric : metrics) {
                metric.addResult(expectedOne, actualOne);
            }
        }
    }

    public TestMetric[] getMetrics(){return this.metrics;}

}
