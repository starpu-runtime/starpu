/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/** @file
 * @brief Listes doublement chainées automatiques
 */


/** @remarks list how-to
 * *********************************************************
 * LIST_TYPE(FOO, contenu);
 *  - déclare les types suivants
 *      + pour les cellules : FOO_t
 *      + pour les listes : FOO_list_t
 *      + pour les itérateurs : FOO_itor_t
 *  - déclare les accesseurs suivants :
 *     * création d'une cellule 
 *   FOO_t      FOO_new(void);  
 *     * suppression d'une cellule
 *   void       FOO_delete(FOO_t); 
 *     * création d'une liste (vide)
 *   FOO_list_t FOO_list_new(void);
 *     * suppression d'une liste
 *   void       FOO_list_delete(FOO_list_t);
 *     * teste si une liste est vide
 *   int        FOO_list_empty(FOO_list_t);
 *     * retire un élément de la liste
 *   void       FOO_list_erase(FOO_list_t, FOO_t);
 *     * ajoute une élément en queue de liste
 *   void       FOO_list_push_back(FOO_list_t, FOO_t);
 *     * ajoute un élément en tête de list
 *   void       FOO_list_push_front(FOO_list_t, FOO_t);
 *     * ajoute la deuxième liste à la fin de la première liste
 *   FOO_t      FOO_list_push_list_back(FOO_list_t, FOO_list_t);
 *     * ajoute la première liste au début de la deuxième liste
 *   FOO_t      FOO_list_push_list_front(FOO_list_t, FOO_list_t);
 *     * retire l'élément en queue de liste
 *   FOO_t      FOO_list_pop_back(FOO_list_t);
 *     * retire l'élement en tête de liste
 *   FOO_t      FOO_list_pop_front(FOO_list_t);
 *     * retourne l'élément en queue de liste
 *   FOO_t      FOO_list_back(FOO_list_t);
 *     * retourne l'élement en tête de liste
 *   FOO_t      FOO_list_front(FOO_list_t);
 *     * vérifie si la liste chainée est cohérente
 *   int	FOO_list_check(FOO_list_t);
 * *********************************************************
 * Exemples d'utilisation :
 *  - au départ, on a :
 *    struct ma_structure_s
 *    {
 *      int a;
 *      int b;
 *    };
 *  - on veut en faire une liste. On remplace la déclaration par :
 *    LIST_TYPE(ma_structure,
 *      int a;
 *      int b;
 *    );
 *    qui crée les types ma_structure_t et ma_structure_list_t.
 *  - allocation d'une liste vide :
 *  ma_structure_list_t l = ma_structure_list_new();
 *  - ajouter un élément 'e' en tête de la liste 'l' :
 *  ma_structure_t e = ma_structure_new();
 *  e->a = 0;
 *  e->b = 1;
 *  ma_structure_list_push_front(l, e);
 *  - itérateur de liste :
 *  ma_structure_itor_t i;
 *  for(i  = ma_structure_list_begin(l);
 *      i != ma_structure_list_end(l);
 *      i  = ma_structure_list_next(i))
 *  {
 *    printf("a=%d; b=%d\n", i->a, i->b);
 *  }
 * *********************************************************
 */



/**@hideinitializer
 * Generates a new type for list of elements */
#define LIST_TYPE(ENAME, DECL) \
  LIST_DECLARE_TYPE(ENAME) \
  LIST_CREATE_TYPE(ENAME, DECL)

/**@hideinitializer
 * Forward type declaration for lists */
#define LIST_DECLARE_TYPE(ENAME) \
  /** automatic type: ENAME##_list_t is a list of ENAME##_t */ \
  typedef struct ENAME##_list_s* ENAME##_list_t; \
  /** automatic type: defines ENAME##_t */ \
  typedef struct ENAME##_s* ENAME##_t; \
  /** automatic type: ENAME##_itor_t is an iterator on lists of ENAME##_t */ \
  typedef ENAME##_t ENAME##_itor_t;

/**@hideinitializer
 * The effective type declaration for lists */
#define LIST_CREATE_TYPE(ENAME, DECL) \
  /** from automatic type: ENAME##_t */ \
  struct ENAME##_s \
  { \
    struct ENAME##_s*_prev; /**< @internal previous cell */ \
    struct ENAME##_s*_next; /**< @internal next cell */ \
    DECL \
  }; \
  /** @internal */ \
  struct ENAME##_list_s \
  { \
    struct ENAME##_s* _head; /**< @internal head of the list */ \
    struct ENAME##_s* _tail; /**< @internal tail of the list */ \
  }; \
  /** @internal */static inline ENAME##_t ENAME##_new(void) \
    { ENAME##_t e = (ENAME##_t)malloc(sizeof(struct ENAME##_s)); \
      e->_next = NULL; e->_prev = NULL; return e; } \
  /** @internal */static inline void ENAME##_delete(ENAME##_t e) \
    { free(e); } \
  /** @internal */static inline void ENAME##_list_push_front(ENAME##_list_t l, ENAME##_t e) \
    { if(l->_tail == NULL) l->_tail = e; else l->_head->_prev = e; \
      e->_prev = NULL; e->_next = l->_head; l->_head = e; } \
  /** @internal */static inline void ENAME##_list_push_back(ENAME##_list_t l, ENAME##_t e) \
    { if(l->_head == NULL) l->_head = e; else l->_tail->_next = e; \
      e->_next = NULL; e->_prev = l->_tail; l->_tail = e; } \
  /** @internal */static inline void ENAME##_list_push_list_front(ENAME##_list_t l1, ENAME##_list_t l2) \
    { if (l2->_head == NULL) { l2->_head = l1->_head; l2->_tail = l1->_tail; } \
      else if (l1->_head != NULL) { l1->_tail->_next = l2->_head; l2->_head->_prev = l1->_tail; l2->_head = l1->_head; } } \
  /** @internal */static inline void ENAME##_list_push_list_back(ENAME##_list_t l1, ENAME##_list_t l2) \
    { if(l1->_head == NULL) { l1->_head = l2->_head; l1->_tail = l2->_tail; } \
      else if (l2->_head != NULL) { l1->_tail->_next = l2->_head; l2->_head->_prev = l1->_tail; l1->_tail = l2->_head; } } \
  /** @internal */static inline ENAME##_t ENAME##_list_front(ENAME##_list_t l) \
    { return l->_head; } \
  /** @internal */static inline ENAME##_t ENAME##_list_back(ENAME##_list_t l) \
    { return l->_tail; } \
  /** @internal */static inline ENAME##_list_t ENAME##_list_new(void) \
    { ENAME##_list_t l; l=(ENAME##_list_t)malloc(sizeof(struct ENAME##_list_s)); \
      l->_head=NULL; l->_tail=l->_head; return l; } \
  /** @internal */static inline int ENAME##_list_empty(ENAME##_list_t l) \
    { return (l->_head == NULL); } \
  /** @internal */static inline void ENAME##_list_delete(ENAME##_list_t l) \
    { free(l); } \
  /** @internal */static inline void ENAME##_list_erase(ENAME##_list_t l, ENAME##_t c) \
    { ENAME##_t p = c->_prev; if(p) p->_next = c->_next; else l->_head = c->_next; \
      if(c->_next) c->_next->_prev = p; else l->_tail = p; } \
  /** @internal */static inline ENAME##_t ENAME##_list_pop_front(ENAME##_list_t l) \
    { ENAME##_t e = ENAME##_list_front(l); \
      ENAME##_list_erase(l, e); return e; } \
  /** @internal */static inline ENAME##_t ENAME##_list_pop_back(ENAME##_list_t l) \
    { ENAME##_t e = ENAME##_list_back(l); \
      ENAME##_list_erase(l, e); return e; } \
  /** @internal */static inline ENAME##_itor_t ENAME##_list_begin(ENAME##_list_t l) \
    { return l->_head; } \
  /** @internal */static inline ENAME##_itor_t ENAME##_list_end(ENAME##_list_t l __attribute__ ((unused))) \
    { return NULL; } \
  /** @internal */static inline ENAME##_itor_t ENAME##_list_next(ENAME##_itor_t i) \
    { return i->_next; } \
  /** @internal */static inline int ENAME##_list_size(ENAME##_list_t l) \
    { ENAME##_itor_t i=l->_head; int k=0; while(i!=NULL){k++;i=i->_next;} return k; } \
  /** @internal */static inline int ENAME##_list_check(ENAME##_list_t l) \
    { ENAME##_itor_t i=l->_head; while(i) \
    { if ((i->_next == NULL) && i != l->_tail) return 0; \
      if (i->_next == i) return 0; \
      i=i->_next;} return 1; }


