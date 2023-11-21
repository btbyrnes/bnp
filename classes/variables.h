
namespace dpm {

class Parameter {
    public:
        virtual float proposal();

    private:
        float p;
};

class Normal {
    public:
        Normal();
        Normal(float mu);
        Normal(float mu, float sigma);

        void set_mu(float a);
        void set_sigma(float a);
        void set_mu_sigma(float a, float b);

        float generate_proposal();

    // overloads

    // private members
    private:
        float mu;
        float sigma;
};


}