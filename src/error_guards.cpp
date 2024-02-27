#include <stdlib.h>
#include <stdio.h>

#include "error_guards.h"

void CheckAlloc(void* ptr, const char* blame) {
    if (ptr == NULL) {
        printf("malloc failed: %s\n", blame);
        exit(-1);
    }
}
