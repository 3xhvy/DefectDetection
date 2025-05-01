#include <stdio.h>
#include <stdlib.h>

int main() {
    char *buf = NULL;
    size_t size = 0;
    
    // Use-after-free vulnerability
    buf = (char *)malloc(10);
    free(buf);
    // Accessing freed memory
    buf[0] = 'A';
    printf("%c\n", buf[0]);
    
    return 0;
}
