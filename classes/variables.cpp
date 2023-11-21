#include "variables.h"

using namespace dpm;

class Normal {
    public:
        Normal() { mu = 0.0; sigma = 1.0; }
        Normal(float a) { mu = a; sigma = 1.0; }
        Normal(float a, float b) { mu = a; sigma = b; };

        void set_mu(float a) { mu = a; };
        void set_sigma(float b) { sigma = b; };
        void set_mu_sigma(float a, float b) { mu = a; sigma = b; };

        float generate_proposal() { return 0.0; } ;

    private:
        float mu;
        float sigma;
};
