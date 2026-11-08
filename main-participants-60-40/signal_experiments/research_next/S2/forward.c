/* Exact binary shift-register likelihood with fixed emissions. */
#include <math.h>
#include <stdint.h>
#include <string.h>

void exact_forward(const int8_t *observed, int cutoff, int a, int b,
                   int current_polarity, int flip, double low, double high,
                   double marginal, double *log_likelihood, double *next_win) {
    /* Bit zero is the newest tick. Pre-sample bits do not affect warmup. */
    int size = 1 << b;
    int mask = size - 1;
    double probability[2][256];
    for (int state = 0; state < size; ++state) {
        int differ = ((state >> (a - 1)) ^ (state >> (b - 1))) & 1;
        probability[0][state] = differ == current_polarity ? high : low;
        probability[1][state] = differ == current_polarity ? low : high;
    }

    double state_mass[256] = {1.0};
    double next_mass[256];
    double ll = 0.0;
    int known = b;
    int state = 0;
    for (int tick = 0; tick < cutoff; ++tick) {
        int outcome = observed[tick];
        const double *emission = probability[flip >= 0 && tick < flip];
        if (known == b) {
            double p = tick < b ? marginal : emission[state];
            if (outcome >= 0) {
                ll += outcome ? log(p) : log1p(-p);
                state = ((state << 1) & mask) | outcome;
                continue;
            }
            int successor = (state << 1) & mask;
            memset(state_mass, 0, size * sizeof(double));
            state_mass[successor] = 1.0 - p;
            state_mass[successor | 1] = p;
            known = 0;
            state = successor;
            continue;
        }

        double norm = 0.0;
        memset(next_mass, 0, size * sizeof(double));
        for (int previous = 0; previous < size; ++previous) {
            double mass = state_mass[previous];
            if (mass == 0.0) {
                continue;
            }
            int successor = (previous << 1) & mask;
            double p = tick < b ? marginal : emission[previous];
            if (outcome != 1) {
                next_mass[successor] += mass * (1.0 - p);
                norm += mass * (1.0 - p);
            }
            if (outcome != 0) {
                next_mass[successor | 1] += mass * p;
                norm += mass * p;
            }
        }
        if (outcome >= 0) {
            ll += log(norm);
            known++;
            state = ((state << 1) & mask) | outcome;
        } else {
            known = 0;
        }
        for (int i = 0; i < size; ++i) {
            state_mass[i] = next_mass[i] / norm;
        }
    }
    *log_likelihood = ll;
    if (cutoff < b) {
        *next_win = marginal;
    } else if (known == b) {
        *next_win = probability[0][state];
    } else {
        *next_win = 0.0;
        for (int i = 0; i < size; ++i) {
            *next_win += state_mass[i] * probability[0][i];
        }
    }
}

/* Reuse the transition code across hypotheses without a Python call per row. */
void exact_batch(const int8_t *history, int rows, int width, const int32_t *cutoffs,
                 const double *low, const double *high, const double *marginal,
                 int a, int b, int polarity, int flip_age, double *ll, double *next_win) {
    for (int row = 0; row < rows; ++row) {
        int flip = flip_age < 0 ? -1 : cutoffs[row] - flip_age;
        exact_forward(history + row * width, cutoffs[row], a, b, polarity, flip,
                      low[row], high[row], marginal[row], ll + row, next_win + row);
    }
}
