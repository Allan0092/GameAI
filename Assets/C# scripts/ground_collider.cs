using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ground_collider : MonoBehaviour
{
    private BoxCollider2D boxcollidercomponent;
    private SpriteRenderer spriterenderercomponent;

    // Start is called before the first frame update
    void Start()
    {
        boxcollidercomponent = GetComponent<BoxCollider2D>();
        spriterenderercomponent = GetComponent<SpriteRenderer>();

        // if (spriterenderercomponent != null)
        // {
        //     boxcollidercomponent.size = spriterenderercomponent.size;
        // }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
