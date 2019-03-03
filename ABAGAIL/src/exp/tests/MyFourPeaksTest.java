package exp.tests;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.ConvergenceTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class MyFourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        int iter = 200000;
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
        Long start = System.nanoTime();
        fit.train();
        Long end = System.nanoTime();
        Double testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + "; Iterations per sec: " + iter/(testingTime));


        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        iter = 200000;
        fit = new FixedIterationTrainer(sa, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + "; Iterations per sec: " + iter/(testingTime));

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        iter = 200000;
        fit = new FixedIterationTrainer(ga, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("GA: " + ef.value(ga.getOptimal()) + "; Iterations per sec: " + iter/(testingTime));

        MIMIC mimic = new MIMIC(200, 20, pop);
        iter = 20000;
        fit = new FixedIterationTrainer(mimic, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + "; Iterations per sec: " + iter/(testingTime));
    }
}
