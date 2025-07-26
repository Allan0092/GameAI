using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class enemy_hitpoints_system : MonoBehaviour
{
    public int hitpoints = 3;

    // Start is called before the first frame update
    void Start()
    {
        // Note: 'gameObject' is a property of 'MonoBehaviour' that refers to the GameObject to which the script is 
        // attached. You can then use this reference for various operations or modifications on the attached 
        // GameObject.

        // This is how you create a reference to the GameObject this C# script is attached to.
                // GameObject thisGameObject = gameObject;
    }

    // Update is called once per frame
    void Update()
    {
        if (hitpoints <= 0)
        {
            Destroy(gameObject);
            print("An enemy is killed!");
        }
    }

    void OnCollisionEnter2D(Collision2D other)
    {
        // Check if the collided object has the tag 'arrow'
        if (other.gameObject.CompareTag("arrow"))
        {
            // Code to handle collision with arrow
            // For example, decrease hitpoints or trigger damage animation
            TakeDamage();
        }
    }

    void TakeDamage()
    {
        hitpoints--;
        print("An enemy took damage!");
    }
}
