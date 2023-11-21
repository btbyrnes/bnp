#include <iostream>
#include <vector>

#include "variables.h"

using namespace std;


class Data {
    public:
        void read_stream(istream& os);
        vector<float> get_y();
        vector<float> get_s();

    private:
        vector<float> y;
        vector<int> s;
};



void get_vector_from_stream(vector<int>& v) {
    for (int i = 0; i < 10; i++) {
        v.push_back(i);
    }
};

int main() {

    vector<int> v;

    get_vector_from_stream(v);

    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << endl;
    }

    cout << "success" << endl;
}

