package exp.tests;

import opt.HillClimbingProblem;
import opt.OptimizationAlgorithm;
import shared.Instance;

/**
 * A randomized hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class MyRandomizedHillClimbing extends OptimizationAlgorithm {

    /**
     * The current optimization data
     */
    private Instance cur;

    private Instance currentBest;
    /**
     * The current value of the data
     */
    private double curVal;

    private double curBestVal;

    private int numRepeats = 0;

    private int repeatThresh = 10;

    /**
     * Make a new randomized hill climbing
     */
    public MyRandomizedHillClimbing(HillClimbingProblem hcp) {
        super(hcp);
        cur = hcp.random();
        curVal = hcp.value(cur);
        currentBest = new Instance(cur.getData());
        curBestVal = new Double(curVal);
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        Instance neigh = hcp.neighbor(cur);
        double neighVal = hcp.value(neigh);

        if (neighVal > this.curVal) {
            curVal = new Double(neighVal);
            cur = (Instance) neigh.copy();
            numRepeats = 0;
            if(neighVal > this.curBestVal){
                currentBest = (Instance) neigh.copy();
                curBestVal = new Double(neighVal);
            }

        }else if (neighVal <= this.curVal){
            numRepeats ++;
            if(numRepeats > repeatThresh){
                System.out.println("restart");
                Instance ne = hcp.random();
                cur = ne;
                curVal  = hcp.value(cur);
                numRepeats = 0;
            }
        }
        return curVal;
    }

    /**
     * @see OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        return currentBest;
    }
    /**
     * @see OptimizationAlgorithm#getOptimalData()
     */

}
