CC = gcc
CFLAGS = -Wall -Wextra -Werror -Iinclude
LDFLAGS =
SRCDIR = src
OBJDIR = build/obj
BINDIR = build/bin
TARGET = $(BINDIR)/cml
SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Compile
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)