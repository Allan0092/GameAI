using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class enemy_taking_damage_from_projectile : MonoBehaviour
{
    enemy_hitpoints_system enemy_hitpoints_system_script_reference;

    // Start is called before the first frame update
    void Start()
    {
        enemy_hitpoints_system_script_reference = GetComponent<enemy_hitpoints_system>();  
    }

    // Update is called once per frame
    void Update()
    {
 
    }

    // OnCollisionEnter2D is called when another collider collides with this clone projectile GameObject

    // 'Collision2D other': The parameter 'other' is an instance of the Collision2D class, which provides information 
    // about the collision.
    void OnCollisionEnter2D(Collision2D other)
    {
        
        // 'if' (other.gameObject.tag == "enemy"): This checks if the GameObject that the current GameObject collided with has a 
        // tag "enemy." The tag is a way to categorize GameObjects in Unity.
        if (other.gameObject.CompareTag("arrow"))
        {
            print("Projectile collision with enemy detected");
            
            
            if (enemy_hitpoints_system_script_reference != null)
            {
                enemy_hitpoints_system_script_reference.hitpoints = enemy_hitpoints_system_script_reference.hitpoints - 1;
            }
        }
    }
}
