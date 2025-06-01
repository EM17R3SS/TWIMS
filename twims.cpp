#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace std;

vector<double> sampleData = {
    -7.009, -8.417, -3.442, -1.079, -5.927, -2.064, -3.617, -4.671, -0.715,
    -3.527, -3.309, 4.677,  -0.091, -5.819, 3.903,  2.184,  -0.049, -5.337,
    -7.568, -8.148, -1.225, -3.129, 0.659,  -1.329, -0.242, -2.619, 0.509,
    -6.320, -3.970, -3.317, -2.265, -8.028, -1.777, -3.330, -2.227, -2.603,
    -5.877, -3.280, -3.121, 1.719,  -0.411, -5.182, -5.352, -0.923, 0.162,
    0.083,  -6.737, -6.380, -4.612, -5.870};

const double alpha = 0.20;
const double c = -3.60;
const double d = 0.00;
const double h = 1.20;
const double a0 = -3.00;
const double sigma0 = 3.00;
const double a1 = -13.00;
const double sigma1 = 3.00;

double calculateMean(const vector<double> &data) {
  return accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double calculateVariance(const vector<double> &data, double meanVal) {
  double sum = 0.0;
  for (double x : data) {
    sum += (x - meanVal) * (x - meanVal);
  }
  return sum / (data.size() - 1);
}

double calculateMedian(vector<double> data) {
  sort(data.begin(), data.end());
  size_t n = data.size();
  if (n % 2 == 0) {
    return (data[n / 2 - 1] + data[n / 2]) / 2.0;
  } else {
    return data[n / 2];
  }
}

double calculateSkewness(const vector<double> &data, double meanVal,
                         double stdDev) {
  double sum = 0.0;
  for (double x : data) {
    sum += pow((x - meanVal) / stdDev, 3);
  }
  return sum / data.size();
}

double calculateKurtosis(const vector<double> &data, double meanVal,
                         double stdDev) {
  double sum = 0.0;
  for (double x : data) {
    sum += pow((x - meanVal) / stdDev, 4);
  }
  return (sum / data.size()) - 3.0;
}

double calculateProbabilityInInterval(const vector<double> &data, double c,
                                      double d) {
  int count = 0;
  for (double x : data) {
    if (x >= c && x <= d) {
      count++;
    }
  }
  return static_cast<double>(count) / data.size();
}

vector<double> buildVariationSeries(const vector<double> &data) {
  vector<double> series = data;
  sort(series.begin(), series.end());
  return series;
}

map<pair<double, double>, int> buildHistogram(const vector<double> &data,
                                              double h) {
  map<pair<double, double>, int> histogram;
  double minVal = *min_element(data.begin(), data.end());
  double maxVal = *max_element(data.begin(), data.end());

  double lower = floor(minVal / h) * h;
  double upper = ceil(maxVal / h) * h;

  for (double binStart = lower; binStart < upper; binStart += h) {
    double binEnd = binStart + h;
    int count = 0;
    for (double x : data) {
      if (x >= binStart && x < binEnd) {
        count++;
      }
    }
    histogram[{binStart, binEnd}] = count;
  }

  return histogram;
}

void performKolmogorovTest(const vector<double> &data, double a, double sigma) {
  vector<double> sortedData = buildVariationSeries(data);
  double n = data.size();
  double D = 0.0;

  boost::math::normal_distribution<> norm(a, sigma);

  for (size_t i = 0; i < sortedData.size(); ++i) {
    double F = boost::math::cdf(norm, sortedData[i]);
    double Dplus = (i + 1) / n - F;
    double Dminus = F - i / n;
    D = max(D, max(Dplus, Dminus));
  }

  double Dcritical = 1.07 / sqrt(n);

  cout << "Kolmogorov test:" << endl;
  cout << "D = " << D << endl;
  cout << "D critical approx " << Dcritical << endl;
  if (D > Dcritical) {
    cout << "Reject H0 at level " << alpha << endl;
  } else {
    cout << "Do not reject H0 at level " << alpha << endl;
  }
}

void performChiSquaredTest(const map<pair<double, double>, int> &histogram,
                           double a, double sigma,
                           bool complexHypothesis = false) {
  int k = histogram.size();
  double n = sampleData.size();
  double chiSq = 0.0;

  boost::math::normal_distribution<> norm(a, sigma);

  int r = complexHypothesis ? 2 : 0;
  int df = k - 1 - r;

  for (const auto &bin : histogram) {
    double p = boost::math::cdf(norm, bin.first.second) -
               boost::math::cdf(norm, bin.first.first);
    double expected = p * n;
    if (expected < 5)
      continue;
    chiSq += pow(bin.second - expected, 2) / expected;
  }

  boost::math::chi_squared_distribution<> chiDist(df);
  double chiCritical = boost::math::quantile(chiDist, 1.0 - alpha);

  cout << (complexHypothesis ? "Chi-squared test (complex hypothesis):"
                             : "Chi-squared test:")
       << endl;
  cout << "chi squared = " << chiSq << endl;
  cout << "chi critical approx " << chiCritical << endl;
  if (chiSq > chiCritical) {
    cout << "Reject H0 at level " << alpha << endl;
  } else {
    cout << "Do not reject H0 at level " << alpha << endl;
  }
}

int main() {
  vector<double> variationSeries = buildVariationSeries(sampleData);
  map<pair<double, double>, int> histogram = buildHistogram(sampleData, h);

  cout << "Variation series:" << endl;
  for (double x : variationSeries) {
    cout << x << " ";
  }
  cout << endl << endl;

  cout << "Histogram (interval -> frequency):" << endl;
  for (const auto &bin : histogram) {
    cout << "[" << bin.first.first << ", " << bin.first.second
         << "): " << bin.second << endl;
  }
  cout << endl;

  double meanVal = calculateMean(sampleData);
  double var = calculateVariance(sampleData, meanVal);
  double stdDev = sqrt(var);
  double med = calculateMedian(sampleData);
  double skew = calculateSkewness(sampleData, meanVal, stdDev);
  double kurt = calculateKurtosis(sampleData, meanVal, stdDev);
  double prob = calculateProbabilityInInterval(sampleData, c, d);

  cout << fixed << setprecision(6);
  cout << "Sample characteristics:" << endl;
  cout << "a) Mean: " << meanVal << endl;
  cout << "b) Variance: " << var << endl;
  cout << "c) Std Dev: " << stdDev << endl;
  cout << "d) Median: " << med << endl;
  cout << "e) Skewness: " << skew << endl;
  cout << "f) Kurtosis: " << kurt << endl;
  cout << "g) P(X in [" << c << ", " << d << "]): " << prob << endl << endl;

  double mleMean = meanVal;
  double mleVar = var * (sampleData.size() - 1) / sampleData.size();
  double mmMean = meanVal;
  double mmVar = var;

  cout << "Parameter estimates:" << endl;
  cout << "MLE: a = " << mleMean << ", sigma squared = " << mleVar << endl;
  cout << "Method of moments: a = " << mmMean << ", sigma squared = " << mmVar
       << endl
       << endl;

  boost::math::students_t_distribution<> tDist(sampleData.size() - 1);
  double t = boost::math::quantile(tDist, 1.0 - alpha / 2.0);
  double meanCiLower = meanVal - t * stdDev / sqrt(sampleData.size());
  double meanCiUpper = meanVal + t * stdDev / sqrt(sampleData.size());

  boost::math::chi_squared_distribution<> chiDist(sampleData.size() - 1);
  double chiLower = boost::math::quantile(chiDist, alpha / 2.0);
  double chiUpper = boost::math::quantile(chiDist, 1.0 - alpha / 2.0);
  double varCiLower = (sampleData.size() - 1) * var / chiUpper;
  double varCiUpper = (sampleData.size() - 1) * var / chiLower;

  cout << "Confidence intervals (level " << 1 - alpha << "):" << endl;
  cout << "For mean: [" << meanCiLower << ", " << meanCiUpper << "]" << endl;
  cout << "For variance: [" << varCiLower << ", " << varCiUpper << "]" << endl
       << endl;

  performKolmogorovTest(sampleData, a0, sigma0);
  cout << endl;

  performChiSquaredTest(histogram, a0, sigma0);
  cout << endl;

  performChiSquaredTest(histogram, meanVal, stdDev, true);
  cout << endl;

  return 0;
}
