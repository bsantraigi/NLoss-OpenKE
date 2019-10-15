#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>
#include <random>

unsigned long long *next_random;

extern "C"
void randReset() {
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	for (INT i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

//[a,b)
INT rand(INT a, INT b){
	return (rand() % (b-a))+ a;
}

using namespace std;
class Uniform_Dist{
public:
   static Uniform_Dist* Instance();
   INT sample();
   void init(INT s, INT e);

private:
	Uniform_Dist(){};  // Private so that it can  not be called
	Uniform_Dist(Uniform_Dist const&){};             // copy constructor is private
	Uniform_Dist& operator=(Uniform_Dist const&){};  // assignment operator is private
	static Uniform_Dist* m_pInstance;

	std::random_device rand_dev;
	std::mt19937 generator;
	std::uniform_int_distribution<int> distr;
};
Uniform_Dist* Uniform_Dist::m_pInstance = NULL;

Uniform_Dist* Uniform_Dist::Instance(){
	if(!m_pInstance){
		m_pInstance = new Uniform_Dist;
	}
	return m_pInstance;
}

void Uniform_Dist::init(INT s, INT e){
    generator = std::mt19937(rand_dev());
	distr = std::uniform_int_distribution<int>(s,e);
}

INT Uniform_Dist::sample(){
	return distr(generator);
}

#endif
