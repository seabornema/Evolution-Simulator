#pragma once
// Thanks to Nikita Lisitsa for the original code
// https://lisyarus.github.io/blog/posts/building-a-quadtree.html

#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <header/creature.h>


using node_id = std::uint32_t;
static constexpr node_id null = node_id(-1);


inline Vec2 operator*(float s, Vec2 const & v) { return v * s; }

inline Vec2 middle(Vec2 const & p1, Vec2 const & p2)
{
    return (p1 + p2) * 0.5f;
}


struct box
{
    Vec2 min{  std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity() };
    Vec2 max{ -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity() };

    box & operator|=(Vec2 const & p)
    {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        return *this;
    }
};

template <typename Iterator>
box bbox(Iterator begin, Iterator end)
{
    box result;
    for (auto it = begin; it != end; ++it)
        result |= it->pos;
    return result;
}


struct node
{
    node_id children[2][2] {
        { null, null },
        { null, null }
    };
};


struct quadtree
{
    box            bbox;
    node_id        root;
    std::vector<node> nodes;
};


static constexpr int MAX_DEPTH = 16; 

template <typename Iterator>
node_id build_impl(quadtree & tree, box const & bbox,
                   Iterator begin, Iterator end, int depth = 0)
{
    if (begin == end) return null;   // nothing to do

    node_id result = static_cast<node_id>(tree.nodes.size());
    tree.nodes.emplace_back();       // push an empty node

    if (begin + 1 == end) return result;   // leaf
    if (depth >= MAX_DEPTH) return result;

    Vec2 center = middle(bbox.min, bbox.max);

    auto bottom = [&center](Creature const & c){ return c.pos.y < center.y; };
    auto left   = [&center](Creature const & c){ return c.pos.x < center.x; };

    Iterator split_y       = std::partition(begin,   end,     bottom);
    Iterator split_x_lower = std::partition(begin,   split_y, left);
    Iterator split_x_upper = std::partition(split_y, end,     left);

    // [bottom/top][left/right]
    tree.nodes[result].children[0][0] = build_impl(tree,
        { bbox.min, center },
        begin, split_x_lower, depth + 1);

    tree.nodes[result].children[0][1] = build_impl(tree,
        { { center.x, bbox.min.y }, { bbox.max.x, center.y } },
        split_x_lower, split_y, depth + 1);

    tree.nodes[result].children[1][0] = build_impl(tree,
        { { bbox.min.x, center.y }, { center.x, bbox.max.y } },
        split_y, split_x_upper, depth + 1);

    tree.nodes[result].children[1][1] = build_impl(tree,
        { center, bbox.max },
        split_x_upper, end, depth + 1);

    return result;
}


  template <typename Iterator, typename Callback>
void query_impl(quadtree const & tree, node_id node,
                box const & bbox, Iterator begin, Iterator end,
                Vec2 const & center, float radius, Callback && cb)
{
    if (node == null || begin == end) return;

 
    Vec2 clamped{
        std::max(bbox.min.x, std::min(center.x, bbox.max.x)),
        std::max(bbox.min.y, std::min(center.y, bbox.max.y))
    };
    Vec2 diff{ clamped.x - center.x, clamped.y - center.y };
    if (diff.x * diff.x + diff.y * diff.y > radius * radius) return;

  
    if (begin + 1 == end ||
        (tree.nodes[node].children[0][0] == null &&
         tree.nodes[node].children[0][1] == null &&
         tree.nodes[node].children[1][0] == null &&
         tree.nodes[node].children[1][1] == null))
    {
        for (auto it = begin; it != end; ++it)
        {
            Vec2 d{ it->pos.x - center.x, it->pos.y - center.y };
            if (d.x * d.x + d.y * d.y <= radius * radius)
                cb(*it);
        }
        return;
    }

  
    Vec2 mid = middle(bbox.min, bbox.max);

    auto bottom        = [&mid](Creature const & c){ return c.pos.y < mid.y; };
    auto left          = [&mid](Creature const & c){ return c.pos.x < mid.x; };
    Iterator split_y       = std::partition(begin,   end,     bottom);
    Iterator split_x_lower = std::partition(begin,   split_y, left);
    Iterator split_x_upper = std::partition(split_y, end,     left);

    query_impl(tree, tree.nodes[node].children[0][0],
               { bbox.min, mid },
               begin, split_x_lower, center, radius, cb);
    query_impl(tree, tree.nodes[node].children[0][1],
               { { mid.x, bbox.min.y }, { bbox.max.x, mid.y } },
               split_x_lower, split_y, center, radius, cb);
    query_impl(tree, tree.nodes[node].children[1][0],
               { { bbox.min.x, mid.y }, { mid.x, bbox.max.y } },
               split_y, split_x_upper, center, radius, cb);
    query_impl(tree, tree.nodes[node].children[1][1],
               { mid, bbox.max },
               split_x_upper, end, center, radius, cb);
}

template <typename Iterator, typename Callback>
void query(quadtree const & tree, Iterator begin, Iterator end,
           Vec2 const & center, float radius, Callback && cb)
{
    query_impl(tree, tree.root, tree.bbox, begin, end, center, radius, cb);
}
template <typename Iterator>
quadtree build(Iterator begin, Iterator end)
{
    quadtree result;
    result.bbox = bbox(begin, end);
    result.root = build_impl(result, result.bbox, begin, end);
    return result;
}

template <typename Iterator>
void collide_all(quadtree const & tree, Iterator begin, Iterator end)
{
    for (auto it = begin; it != end; ++it)
    {
        query(tree, begin, end, it->pos, it->radius,
            [&](Creature & other)
            {
                if (&other != &*it)
                    other.collide(*it);
            });
    }
}
