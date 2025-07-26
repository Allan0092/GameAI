using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class character_hitpoints_system : MonoBehaviour
{
    public int hitpoints = 3;
    public string gameOverPanelName = "game_over_text";
    private GameObject gameOverPanel;


    public string hitpointsThreeTextName = "hitpoints_3_text";
    private GameObject hitpointsThreeText;
    public string hitpointsTwoTextName = "hitpoints_2_text";
    private GameObject hitpointsTwoText;
    public string hitpointsOneTextName = "hitpoints_1_text";
    private GameObject hitpointsOneText;
    public string hitpointsZeroTextName = "hitpoints_0_text";
    private GameObject hitpointsZeroText;


    // Start is called before the first frame update
    void Start()
    {
        hitpointsThreeText = GameObject.Find(hitpointsThreeTextName);
        hitpointsTwoText = GameObject.Find(hitpointsTwoTextName);
        hitpointsOneText = GameObject.Find(hitpointsOneTextName);
        hitpointsZeroText = GameObject.Find(hitpointsZeroTextName);

        hitpointsThreeText.SetActive(true);
        hitpointsTwoText.SetActive(false);
        hitpointsOneText.SetActive(false);
        hitpointsZeroText.SetActive(false);


        Vector2 screenCenter = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f);
        Vector2 worldCenter = Camera.main.ScreenToWorldPoint(new Vector3(screenCenter.x, screenCenter.y, 10f));

        hitpointsThreeText.transform.position = worldCenter;
        hitpointsTwoText.transform.position = worldCenter;
        hitpointsOneText.transform.position = worldCenter;
        hitpointsZeroText.transform.position = worldCenter;


        // Note: 'gameObject' is a property of 'MonoBehaviour' that refers to the GameObject to which the script is 
        // attached. You can then use this reference for various operations or modifications on the attached 
        // GameObject.

        // This is how you create a reference to the GameObject this C# script is attached to.
                // GameObject thisGameObject = gameObject;
        gameOverPanel = GameObject.Find(gameOverPanelName);
        
        if (gameOverPanel != null)
        {
            gameOverPanel.SetActive(false);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (hitpoints == 2)
        {
            TwoHitpointsText();
        }

        if (hitpoints == 1)
        {
            OneHitpointsText();
        }

        if (hitpoints <= 0)
        {

            // Destroy(gameObject);
            print("You are killed!");
            ShowGameOver();
            ZeroHitpointsText();
            
            character_movement character_movement_reference_script = GetComponent<character_movement>();
            character_jump_movement character_jump_movement_reference_script = GetComponent<character_jump_movement>();

            character_movement_reference_script.ToggleMovement(false);
            character_jump_movement_reference_script.ToggleJumpMovement(false);


        }
    }

    void OnCollisionEnter2D(Collision2D other)
    {
        // Check if the collided object has the tag 'arrow'
        if (other.gameObject.CompareTag("enemy"))
        {
            // Code to handle collision with arrow
            // For example, decrease hitpoints or trigger damage animation
            TakeDamage();
        }
    }

    void TakeDamage()
    {
        hitpoints--;
        print("You took damage!");
    }

    // If character hitpoints reaches 0
    void ShowGameOver()
    {
        // Activate the Game Over panel
        if (gameOverPanel != null)
        {
            // Set the Game Over panel's position to the center of the screen
            Vector2 screenCenter = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f);
            Vector2 worldCenter = Camera.main.ScreenToWorldPoint(new Vector3(screenCenter.x, screenCenter.y, 10f));

            gameOverPanel.transform.position = worldCenter;

            gameOverPanel.SetActive(true);
        }
        else
        {
            Debug.LogError("Game Over panel reference not set in the inspector.");
        }
    }

    void TwoHitpointsText()
    {
        hitpointsThreeText.SetActive(false);
        hitpointsTwoText.SetActive(true);
    }

    void OneHitpointsText()
    {
        hitpointsTwoText.SetActive(false);
        hitpointsOneText.SetActive(true);
    }

    void ZeroHitpointsText()
    {
        hitpointsOneText.SetActive(false);
        hitpointsZeroText.SetActive(true);
    }
}
