package exp.tests;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class MyTravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        int iter = 200000;
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
        Long start = System.nanoTime();
        fit.train();
        Long end = System.nanoTime();
        Double testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("RHC Optimal Value: " + ef.value(rhc.getOptimal()) + " IterPerSec: " + iter/testingTime );

        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        iter = 200000;
        fit = new FixedIterationTrainer(sa, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + "; IterPerSec: " + iter/(testingTime));

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        iter = 200000;
        fit = new FixedIterationTrainer(ga, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("GA: " + ef.value(ga.getOptimal()) + "; IterPerSec: " + iter/(testingTime));

        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        MIMIC mimic = new MIMIC(200, 100, pop);
        iter = 2000;
        fit = new FixedIterationTrainer(mimic, iter);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        testingTime = new Double(end - start);
        testingTime /= Math.pow(10, 9);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + "; IterPerSec: " + iter/(testingTime));

    }
}
