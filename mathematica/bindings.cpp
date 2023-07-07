#include "wll_interface.h"

double multiply(double x, int y)
{
    return x * y;
}

DEFINE_WLL_FUNCTION(multiply)  // defines wll_multiply