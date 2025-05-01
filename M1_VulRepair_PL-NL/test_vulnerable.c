#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[10];
    // Buffer overflow vulnerability - no bounds checking
    strcpy(buffer, input);
    printf("Buffer contains: %s\n", buffer);
}

int main() {
    char *user_input = "This is a very long string that will cause a buffer overflow";
    vulnerable_function(user_input);
    return 0;
}
