using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class firing_projectiles : MonoBehaviour
{
    public GameObject projectilePrefab;
    public float projectileSpeed = 15f;
    public float maxProjectileDistance = 3f;

    private character_movement characterMovementScriptReference;

    // Getting a reference to the blue chief bird GameObject
    public string chief_bird_name = "blue bird sprite_0";
    public GameObject chief_bird;
    private chief_NPC_interaction chief_NPC_interaction_script_reference;
    
    // Start is called before the first frame update
    void Start()
    {
        chief_bird = GameObject.Find(chief_bird_name);
        
        characterMovementScriptReference = GetComponent<character_movement>();
    }

        
    // Update is called once per frame
    void Update()
    {
        chief_NPC_interaction chief_NPC_interaction_script_reference = chief_bird.GetComponent<chief_NPC_interaction>();

        // Allowing the player to shoot projectiles only if he has the bow/completed interaction with the
        // chief bird, and if the 'S' key is pressed
        if (Input.GetKeyDown(KeyCode.S) && chief_NPC_interaction_script_reference.completedinteraction == true)
        {
            ShootProjectile();
        }
    }

    void ShootProjectile()
    {
        // Instantiate the projectile
        GameObject projectile = Instantiate(projectilePrefab, transform.position, Quaternion.identity);


        // Setting all the projectile clones to be of a higher sorting order (by default this is 0)
        SpriteRenderer projectileRenderer = projectile.GetComponent<SpriteRenderer>();
        projectileRenderer.sortingOrder = 6;

        projectile.tag = "arrow";

        // Access the Rigidbody2D component from the instantiated projectile
        Rigidbody2D projectileRb = projectile.GetComponent<Rigidbody2D>();
        // Access the BoxCollider2D component from the instantiated projectile
        BoxCollider2D projectileBc = projectile.GetComponent<BoxCollider2D>();

        // Check if the Rigidbody2D component is not null
        if (projectileRb == null)
        {
            // If the Rigidbody2D component is not found, add it to the projectile
            projectileRb = projectile.AddComponent<Rigidbody2D>();
            
        }

        // Check if the Rigidbody2D component is not null
        if (projectileBc == null)
        {
            // If the Rigidbody2D component is not found, add it to the projectile
            projectileBc = projectile.AddComponent<BoxCollider2D>();
        }


        // Ignore collision of character sprite with each arrow GameObject
        GameObject[] arrows = GameObject.FindGameObjectsWithTag("arrow");

        foreach (GameObject arrow in arrows)
        {
            Physics2D.IgnoreCollision(gameObject.GetComponent<BoxCollider2D>(), arrow.GetComponent<BoxCollider2D>(), true);
        }



        if (characterMovementScriptReference.isfacingright == true)
        {
            // Apply force to the projectile to make it move in the right direction the player is facing
            projectileRb.linearVelocity = transform.right * (transform.localScale.x > 0 ? projectileSpeed : -projectileSpeed);
        }
        else
        {
            // Apply force to the projectile to make it move in the left direction the player is facing
            projectileRb.linearVelocity = -transform.right * (transform.localScale.x > 0 ? projectileSpeed : -projectileSpeed);
            // Flipping the sprite rendering of the fired clone projectile
            projectileRenderer.flipX = true;
        }

        // Start a coroutine to destroy the projectile after a certain distance
        StartCoroutine(DestroyProjectileAfterDistance(projectile));
    }

    IEnumerator DestroyProjectileAfterDistance(GameObject projectile)
    {
        float traveledDistance = 0f;

        while (traveledDistance < maxProjectileDistance)
        {
            // Calculate the distance traveled
            traveledDistance += projectileSpeed * Time.deltaTime;

            // Wait for the end of the frame
            yield return null;
        }

        // Destroy the projectile after reaching the maximum distance
        Destroy(projectile);

        
    }

}