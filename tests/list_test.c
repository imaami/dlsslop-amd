/** @file
 *
 * The intrusive list of common/list.h: an empty head, static and in place; entries linked at the
 * end and walked in order as their own type, with a hook that is not their first member; an entry
 * unlinked and freed during the walk, first, last and in the middle; list_only() at zero, one and
 * two entries; and container_of() back from a hook.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "list.h"

/** @brief An entry whose hook is not at offset 0. */
struct item {
	double      weight; //!< Something before the hook.
	struct list node;   //!< In a test's list.
	uint32_t    value;  //!< What the walks report.
};

/** @brief Ends the test with a message unless a condition holds. */
[[gnu::format(printf, 2, 3)]]
static void
require (bool        condition,
         char const *fmt,
         ...)
{
	if (condition)
		return;

	va_list args;
	va_start(args, fmt);
	fputs("list-test: ", stderr);
	vfprintf(stderr, fmt, args);
	fputc('\n', stderr);
	va_end(args);
	exit(1);
}

/** @brief Whether a walk of a list meets its entries' values in this order, both ways round. */
static bool
holds (struct list const *head,
       uint32_t const    *values,
       uint32_t           count)
{
	struct item *it;
	uint32_t n = 0;
	list_foreach (it, head, struct item, node) {
		if (n == count || it->value != values[n])
			return false;
		n++;
	}
	if (n != count)
		return false;

	struct list const *node = head->prev;
	for (n = count; n-- > 0; node = node->prev) {
		if (node == head || container_of(node, struct item, node)->value != values[n])
			return false;
	}
	return node == head;
}

/** @brief A static head, as the layer's are. */
static struct list g_list = LIST_INIT(g_list);

int
main (void)
{
	require(g_list.next == &g_list && g_list.prev == &g_list, "a static head does not start empty");
	require(!list_only(&g_list), "an empty list has an only entry");
	require(holds(&g_list, nullptr, 0), "an empty list has entries");

	struct item a = {.value = 1}, b = {.value = 2}, c = {.value = 3};
	list_append(&g_list, &a.node);
	require(list_only(&g_list) == &a.node, "a list of one entry does not return it as its only one");
	require(container_of(list_only(&g_list), struct item, node) == &a, "container_of() misses the entry");
	list_append(&g_list, &b.node);
	require(!list_only(&g_list), "a list of two entries has an only entry");
	list_append(&g_list, &c.node);
	require(holds(&g_list, (uint32_t const[]){1, 2, 3}, 3), "the entries are not in the order they came");

	// The middle entry, unlinked during the walk.
	struct item *it;
	uint32_t visited = 0;
	list_foreach (it, &g_list, struct item, node) {
		visited++;
		if (it == &b)
			list_del(&it->node);
	}
	require(visited == 3, "a walk that unlinked an entry visited %u entries", visited);
	require(!b.node.next && !b.node.prev, "an unlinked hook keeps its pointers");
	require(holds(&g_list, (uint32_t const[]){1, 3}, 2), "unlinking the middle entry broke the list");

	list_del(&a.node);
	list_del(&c.node);
	require(holds(&g_list, nullptr, 0) && !list_only(&g_list), "unlinking every entry left one");

	// Entries on the heap, unlinked and freed during the walk: the first, the last, then the rest.
	struct list head;
	list_init(&head);
	require(holds(&head, nullptr, 0) && !list_only(&head), "an in-place head does not start empty");
	for (uint32_t i = 0; i < 5; i++) {
		struct item *p = calloc(1, sizeof *p);
		require(p, "out of memory");
		p->value = i;
		list_append(&head, &p->node);
	}
	list_foreach (it, &head, struct item, node) {
		if (it->value != 0 && it->value != 4)
			continue;
		list_del(&it->node);
		free(it);
		it = nullptr;
	}
	require(holds(&head, (uint32_t const[]){1, 2, 3}, 3), "freeing the first and the last entry broke the list");
	list_foreach (it, &head, struct item, node) {
		list_del(&it->node);
		free(it);
		it = nullptr;
	}
	require(holds(&head, nullptr, 0), "freeing every entry left one");

	puts("list-test: ok");
	return 0;
}
