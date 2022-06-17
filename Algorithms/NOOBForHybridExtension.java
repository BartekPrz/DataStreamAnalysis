package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.Arrays;

public class NOOBForHybridExtension extends OzaBag {

    private static final long serialVersionUID = 1L;

    protected Instances windowInstances;

    protected NearestNeighbourSearch nearestNeighbourSearch;

    protected double[] classSize;

    protected int numberOfNeighbours;

    protected int windowSize;

    protected double psiCoefficient;

    NOOBForHybridExtension(int windowSize,
                           int numberOfNeighbours,
                           double psiCoefficient,
                           ClassOption baseLearnerOption,
                           IntOption ensembleSizeOption
    ) {
        this.windowSize = windowSize;
        this.numberOfNeighbours = numberOfNeighbours;
        this.psiCoefficient = psiCoefficient;
        this.baseLearnerOption = baseLearnerOption;
        this.ensembleSizeOption = ensembleSizeOption;
    }

    @Override
    public String getPurposeString() {
        return "Neighbourhood oversampling online bagging - part of hybrid extension algorithm";
    }

    @Override
    public void resetLearningImpl() {
        this.windowInstances = null;
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

    private double calculatePoissonLambda(Instance inst) {
        double lambda = 1d;
        int minClass = getMinorityClass();
        double denominator = classSize[minClass] != 0 ? classSize[minClass] : 1;
        double betaCoefficient = classSize[1 - minClass] / denominator;

        if ((int) inst.classValue() == minClass) {
            double safeLevelOfInstance = getSafeLevelOfInstance(inst);
            return betaCoefficient * (safeLevelOfInstance + 1);
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
            if ((int) neighbour.classValue() != inst.classValue())
                numberOfMajorityNeighbours += 1;
        }

        return Math.pow(numberOfMajorityNeighbours, psiCoefficient) / numberOfNeighbours;
    }
}
