CC = gcc
CCPP = g++
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
BUILDDIR := build
SRCDIR := src

all: dir vocab_count cooccur wordcooccurleftfixlen

dir :
	mkdir -p $(BUILDDIR)
cooccur : $(SRCDIR)/cooccur.c
	$(CC) $(SRCDIR)/cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)

vocab_count : $(SRCDIR)/vocab_count.c
	$(CC) $(SRCDIR)/vocab_count.c -o $(BUILDDIR)/vocab_count $(CFLAGS)

wordcooccurleftfixlen: $(SRCDIR)/wordcooccurleftfixlen.c
	$(CC) $(SRCDIR)/wordcooccurleftfixlen.c  -o $(BUILDDIR)/wordcooccurleftfixlen $(CFLAGS)

clean:
	rm -rf vocab_count build cooccur wordcooccurleftfixlen
