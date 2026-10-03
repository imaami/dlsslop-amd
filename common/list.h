/** @file
 *
 * An intrusive circular doubly-linked list with a sentinel head.
 *
 * A struct that lives in a list embeds a struct list, its hook. The list's head is a struct list of
 * its own, which points to itself while the list is empty. Linking and unlinking an entry take
 * constant time and move no other entry, so a pointer to an entry stays valid until the entry's
 * owner frees it. list_foreach() walks the entries as their own type, through container_of().
 */
// SPDX-License-Identifier: MIT
#ifndef DLSSLOP_AMD_COMMON_LIST_H_
#define DLSSLOP_AMD_COMMON_LIST_H_

#include "util.h"

/** @brief A list's head, or an entry's hook in a list. */
struct list {
	struct list *next; //!< The first entry, or the next one; the head after the last entry.
	struct list *prev; //!< The last entry, or the previous one; the head before the first entry.
};

/** @brief The initializer of an empty list's head, also for a head with static storage.
 *
 * @param head The head.
 */
#define LIST_INIT(head) { &(head), &(head) }

/** @brief Makes a head an empty list.
 *
 * @param head The head.
 */
force_inline void
list_init (struct list *head)
{
	head->next = head;
	head->prev = head;
}

/** @brief Links an entry in at the end of a list.
 *
 * @param head The list's head.
 * @param node The entry's hook, which is in no list.
 */
force_inline void
list_append (struct list *head,
             struct list *node)
{
	node->next = head;
	node->prev = head->prev;
	head->prev->next = node;
	head->prev = node;
}

/** @brief Unlinks an entry from its list.
 *
 * @param node The entry's hook, whose pointers are null afterwards.
 */
force_inline void
list_del (struct list *node)
{
	node->next->prev = node->prev;
	node->prev->next = node->next;
	node->next = nullptr;
	node->prev = nullptr;
}

/** @brief A list's only entry.
 *
 * @param head The list's head.
 * @return     The entry's hook if the list has exactly one entry, otherwise nullptr.
 */
force_inline struct list *
list_only (struct list const *head)
{
	return head->next != head && head->next == head->prev ? head->next : nullptr;
}

/** @brief Walks a list's entries from the first to the last.
 *
 * The walk reads an entry's successor before the body runs, so the body may unlink and free the
 * entry. @a head is evaluated at every step.
 *
 * @param entry  A pointer to @a type, which points at each entry in turn.
 * @param head   A pointer to the list's head.
 * @param type   The entries' type.
 * @param member The name of the entries' struct list member.
 */
#define list_foreach(entry, head, type, member) \
	for (struct list *list_node_ = (head)->next, *list_next_ = list_node_->next; \
	     list_node_ != (head) && ((entry) = container_of(list_node_, type, member), true); \
	     list_node_ = list_next_, list_next_ = list_node_->next)

#endif /* DLSSLOP_AMD_COMMON_LIST_H_ */
