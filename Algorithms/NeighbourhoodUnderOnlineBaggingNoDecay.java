package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.Arrays;

public class NeighbourhoodUnderOnlineBaggingNoDecay extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public IntOption numberOfNeighboursOption = new IntOption("neighboursCount", 'k',
            "Number of neighbours to take into account during analysis", 5, 3, Integer.MAX_VALUE);

    public IntOption windowSizeOption = new IntOption("windowSize", 'w',
            "The window size used to analyze the neighbourhood of incoming example", 1000, 1, Integer.MAX_VALUE);

    protected Classifier[] ensemble;

    protected Instances windowInstances;

    protected NearestNeighbourSearch nearestNeighbourSearch;

    protected double classSize[];

    protected int numberOfNeighbours = 0;

    protected int windowSize = 0;

    protected int timestamp = 0;

    @Override
    public String getPurposeString() {
        return "Neighbourhood online bagging for imbalanced data";
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public void resetLearningImpl() {
        this.windowInstances = null;
        this.windowSize = this.windowSizeOption.getValue();
        this.numberOfNeighbours = this.numberOfNeighboursOption.getValue();
        this.nearestNeighbourSearch = new LinearNNSearch();

        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        initVariables();

        this.timestamp += 1;
        updateClassSize(inst);
        double lambda = calculatePoissonLambda(inst);

        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
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
            Arrays.fill(classSize, 1d / classSize.length);
        }

        for (int i = 0; i < classSize.length; ++i) {
            classSize[i] = ((this.timestamp - 1) * classSize[i] + ((int) inst.classValue() == i ? 1d : 0d)) / this.timestamp;
        }
    }

    public double calculatePoissonLambda(Instance inst) {
        double lambda = 1d;
        int minClass = getMinorityClass();
        double betaCoefficient = classSize[minClass] / classSize[1 - minClass];

        if ((int) inst.classValue() != minClass && betaCoefficient != 0) {
            double safeLevelOfInstance = getSafeLevelOfInstance(inst);
            return betaCoefficient * safeLevelOfInstance;
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

    public double getSafeLevelOfInstance(Instance inst) {
        double safeLevel = 1d;
        int numberOfMajorityNeighbours = 0;

        if (windowInstances.size() < numberOfNeighbours) {
            return safeLevel;
        }

        Instances neighbours;
        try {
            nearestNeighbourSearch.setInstances(windowInstances);
            neighbours = nearestNeighbourSearch.kNearestNeighbours(inst, numberOfNeighbours);
        } catch (Exception e) {
            e.printStackTrace();
            return safeLevel;
        }


        for (int i = 0; i < numberOfNeighbours; ++i) {
            Instance neighbour = neighbours.get(i);
            if ((int) neighbour.classValue() == inst.classValue())
                numberOfMajorityNeighbours += 1;
        }

        return (double) numberOfMajorityNeighbours / numberOfNeighbours;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
