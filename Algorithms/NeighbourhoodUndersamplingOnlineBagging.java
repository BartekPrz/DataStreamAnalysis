package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.MiscUtils;

import java.util.Arrays;

public class NeighbourhoodUndersamplingOnlineBagging extends OzaBag {

    private static final long serialVersionUID = 1L;

    public IntOption numberOfNeighboursOption = new IntOption("neighboursCount", 'k',
            "Number of neighbours taken into account during analysis", 5, 3, Integer.MAX_VALUE);

    public IntOption windowSizeOption = new IntOption("windowSize", 'w',
            "The window size used to analyze the neighbourhood of incoming example", 1000, 1, Integer.MAX_VALUE);

    public FloatOption psiCoefficient = new FloatOption("psi", 'p',
            "Additional coefficient for calculating safe level", 1, 0.5, 3);

    protected Instances windowInstances;

    protected NearestNeighbourSearch nearestNeighbourSearch;

    protected double[] classSize;

    protected int numberOfNeighbours = 0;

    protected int windowSize = 0;

    @Override
    public String getPurposeString() {
        return "Neighbourhood undersampling online bagging for imbalanced data";
    }

    @Override
    public void resetLearningImpl() {
        this.windowInstances = null;
        this.windowSize = this.windowSizeOption.getValue();
        this.numberOfNeighbours = this.numberOfNeighboursOption.getValue();
        this.nearestNeighbourSearch = new LinearNNSearch();
        super.resetLearningImpl();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        this.initVariables();

        updateClassSize(inst);
        double lambda = calculatePoissonLambda(inst);

        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }

        if (windowInstances.size() == windowSize) {
            windowInstances.delete(0);
        }
        windowInstances.add(inst);
    }

    private void initVariables() {
        if (this.windowInstances == null) {
            this.windowInstances = new Instances(this.getModelContext());
        }
    }

    protected void updateClassSize(Instance inst) {
        if (this.classSize == null) {
            classSize = new double[inst.numClasses()];

            Arrays.fill(classSize, 0d);
        }

        if (this.windowInstances.size() == windowSize) {
            classSize[(int) this.windowInstances.get(0).classValue()]--;
        }

        classSize[(int) inst.classValue()]++;
    }

    public double calculatePoissonLambda(Instance inst) {
        double lambda = 1d;
        int minClass = getMinorityClass();
        double betaCoefficient = classSize[minClass] / classSize[1 - minClass];

        if ((int) inst.classValue() != minClass) {
            double safeLevelOfInstance = getSafeLevelOfInstance(inst);
            return betaCoefficient * Math.pow(safeLevelOfInstance, psiCoefficient.getValue());
        }

        return lambda;
    }

    public int getMajorityClass() {
        int indexMaj = 0;

        for (int i = 1; i < classSize.length; ++i) {
            if (classSize[i] > classSize[indexMaj]) {
                indexMaj = i;
            }
        }
        return indexMaj;
    }

    public int getMinorityClass() {
        int indexMin = 0;

        for (int i = 1; i < classSize.length; ++i) {
            if (classSize[i] <= classSize[indexMin]) {
                indexMin = i;
            }
        }
        return indexMin;
    }

    private double getSafeLevelOfInstance(Instance inst) {
        int numberOfMajorityNeighbours = 0;
        double safeLevel = 1d;

        Instances neighbours;
        try {
            nearestNeighbourSearch.setInstances(windowInstances);
            neighbours = nearestNeighbourSearch.kNearestNeighbours(inst, numberOfNeighbours);
        } catch (Exception e) {
            e.printStackTrace();
            return safeLevel;
        }

        for (int i = 0; i < neighbours.size(); ++i) {
            Instance neighbour = neighbours.get(i);
            if ((int) neighbour.classValue() == inst.classValue())
                numberOfMajorityNeighbours += 1;
        }

        return (double) numberOfMajorityNeighbours / numberOfNeighbours;
    }
}
